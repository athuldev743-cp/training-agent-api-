
"""
Model-priority cloud LLM chain with optional local fallback.

- Models are attempted in model_registry.py priority order.
- Each model has an independent cooldown.
- Authentication failures temporarily disable that provider.
- HybridLLM falls back to the local model.
- CloudOnlyLLM raises LLMUnavailable when all models fail.
- Streaming and structured requests stay local in HybridLLM.
- Profile FACTS stay local in HybridLLM.
- CloudOnlyLLM retains limitation redaction and non-personal caching.
"""

import collections
import hashlib
import json
import logging
import os
import re as _re
import time
import urllib.error
import urllib.request

from agent.model_registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)
PROVIDERS = tuple(
    (
        config["provider"],
        config["url"],
        config["key_env"],
        None,
        config["model"],
    )
    for config in MODEL_REGISTRY
)


def _post(url, key, body, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "training-agent-api/1.0",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


class LLMUnavailable(RuntimeError):
    """No cloud model produced a valid reply."""


class HybridLLM:
    COOLDOWN_S = 300
    RATE_LIMIT_COOLDOWN_S = 60
    SERVER_ERROR_COOLDOWN_S = 45

    def __init__(
        self,
        local,
        post=_post,
        timeout=8.0,
        env=None,
        deadline=30.0,
        clock=time.monotonic,
    ):
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(override=True)
            except Exception:
                pass

            env = os.environ

        self.local = local
        self._post = post
        self.timeout = timeout
        self.env = env

        self.deadline = deadline
        self.clock = clock

        # Independent cooldowns for models and providers.
        self.skip_until = {}
        self.provider_skip_until = {}

        self.last = {"provider": None, "model": None}
        self.errors = []

    def _cloud_ok(self, kw):
        if self.env.get("CLOUD_LLM") != "1":
            return False

        if kw.get("stream") or kw.get("response_format"):
            return False

        messages = kw.get("messages", [])

        return not any(
            "FACTS:" in str(message.get("content", ""))
            for message in messages
        )

    def _fail(self, model_name, reason, cooldown=None):
        """
        Record failure without storing prompts, API responses or credentials.
        """

        self.errors.append((model_name, reason))

        duration = (
            self.COOLDOWN_S
            if cooldown is None
            else cooldown
        )

        self.skip_until[model_name] = self.clock() + duration

    def _provider_failed(self, provider, reason):
        """
        Provider-wide cooldown for authentication failures.
        """

        self.errors.append((provider, reason))

        self.provider_skip_until[provider] = (
            self.clock() + self.COOLDOWN_S
        )

    def _request_cloud(self, messages, kw, want_json=False):
        """Execute every eligible model in registry priority order."""

        started = self.clock()
        attempted = 0
        skipped = 0

        requested_tokens = kw.get("max_tokens", 160)
        max_tokens = min(800, max(400, 4 * requested_tokens))

        for config in MODEL_REGISTRY:
            model_name = config["name"]
            provider = config["provider"]
            api_key = self.env.get(config["key_env"])

            if not api_key:
                skipped += 1
                logger.warning(
                    "Cloud LLM model skipped provider=%s model=%s reason=missing_api_key",
                    provider, config["model"],
                )
                continue

            now = self.clock()
            model_remaining = self.skip_until.get(model_name, 0) - now

            if model_remaining > 0:
                skipped += 1
                logger.info(
                    "Cloud LLM model skipped provider=%s model=%s "
                    "reason=model_cooldown remaining_s=%s",
                    provider, config["model"], round(model_remaining, 1),
                )
                continue

            provider_remaining = self.provider_skip_until.get(provider, 0) - now

            if provider_remaining > 0:
                skipped += 1
                logger.info(
                    "Cloud LLM model skipped provider=%s model=%s "
                    "reason=provider_cooldown remaining_s=%s",
                    provider, config["model"], round(provider_remaining, 1),
                )
                continue

            remaining = self.deadline - (now - started)

            if remaining <= 0.5:
                skipped += 1
                logger.warning(
                    "Cloud LLM chain deadline reached before provider=%s "
                    "model=%s attempted=%s skipped=%s",
                    provider, config["model"], attempted, skipped,
                )
                break

            model_timeout = min(
                self.timeout,
                config.get("timeout_s", self.timeout),
                remaining,
            )

            body = {
                "model": config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": kw.get("temperature", 0.2),
            }

            if want_json:
                body["response_format"] = {"type": "json_object"}

            attempted += 1
            logger.info(
                "Cloud LLM attempting provider=%s model=%s "
                "timeout_s=%s remaining_s=%s",
                provider, config["model"],
                round(model_timeout, 2), round(remaining, 2),
            )

            try:
                data = self._post(
                    config["url"], api_key, body, model_timeout
                )

                choice = data["choices"][0]
                content = choice["message"]["content"]

                if choice.get("finish_reason") == "length":
                    self._fail(model_name, "truncated")
                    logger.warning(
                        "Cloud LLM rejected provider=%s model=%s reason=truncated",
                        provider, config["model"],
                    )
                    continue

                if not isinstance(content, str) or not content.strip():
                    self._fail(model_name, "empty_response")
                    logger.warning(
                        "Cloud LLM rejected provider=%s model=%s reason=empty_response",
                        provider, config["model"],
                    )
                    continue

                content = content.strip()

                if want_json:
                    json.loads(content)

                result = {
                    "choices": [
                        {"message": {"content": content}}
                    ],
                    "usage": data.get("usage") or {},
                }

                self.last = {
                    "provider": provider,
                    "model": config["model"],
                }

                logger.info(
                    "Cloud LLM model succeeded provider=%s model=%s "
                    "elapsed_ms=%s attempted=%s",
                    provider, config["model"],
                    int((self.clock() - started) * 1000), attempted,
                )

                return result, config

            except json.JSONDecodeError:
                self._fail(model_name, "bad_json")
                logger.warning(
                    "Cloud LLM rejected provider=%s model=%s reason=bad_json",
                    provider, config["model"],
                )

            except Exception as exc:
                status = getattr(exc, "code", None)

                logger.warning(
                    "Cloud LLM attempt failed provider=%s model=%s "
                    "error_type=%s http_status=%s elapsed_ms=%s",
                    provider,
                    config["model"],
                    type(exc).__name__,
                    status if status is not None else "n/a",
                    int((self.clock() - started) * 1000),
                )

                if status in (401, 403):
                    self._fail(model_name, f"http_{status}")
                    self._provider_failed(provider, f"http_{status}")

                elif status == 429:
                    self._fail(
                        model_name, "http_429",
                        self.RATE_LIMIT_COOLDOWN_S,
                    )

                elif isinstance(status, int) and status >= 500:
                    self._fail(
                        model_name, f"http_{status}",
                        self.SERVER_ERROR_COOLDOWN_S,
                    )

                else:
                    self._fail(model_name, type(exc).__name__)

        logger.warning(
            "Cloud LLM chain exhausted attempted=%s skipped=%s "
            "elapsed_ms=%s",
            attempted, skipped,
            int((self.clock() - started) * 1000),
        )

        return None, None

    def create_chat_completion(self, **kw):
        if self._cloud_ok(kw):
            messages = [
                {
                    **message,
                    "content": str(
                        message.get("content", "")
                    ).replace(" /no_think", ""),
                }
                for message in kw["messages"]
            ]

            result, config = self._request_cloud(
                messages,
                kw,
                want_json=False,
            )

            if result is not None:
                return result

        self.last = {
            "provider": "local",
            "model": None,
        }

        return self.local.create_chat_completion(**kw)


class CloudOnlyLLM(HybridLLM):
    """
    Cloud-only model chain.

    Structured JSON requests must return valid JSON.
    Non-personal identical requests are cached.
    """

    CACHE_TTL_S = 6 * 3600
    CACHE_MAX = 500

    def __init__(
        self,
        post=_post,
        timeout=8.0,
        env=None,
        deadline=30.0,
        clock=time.monotonic,
    ):
        super().__init__(
            None,
            post=post,
            timeout=timeout,
            env=env,
            deadline=deadline,
            clock=clock,
        )

        self.cache = collections.OrderedDict()

    def create_chat_completion(self, **kw):
        if kw.get("stream"):
            raise LLMUnavailable(
                "streaming is not supported by the cloud chain"
            )

        messages = [
            {
                **message,
                "content": str(
                    message.get("content", "")
                ).replace(" /no_think", ""),
            }
            for message in kw["messages"]
        ]

        # Keep the existing health-limitation privacy protection.
        if self.env.get("CLOUD_SHARE_LIMITATIONS") != "1":
            messages = [
                {
                    **message,
                    "content": redact(message["content"]),
                }
                for message in messages
            ]

        personal = any(
            "FACTS:" in message["content"]
            for message in messages
        )

        want_json = bool(kw.get("response_format"))

        cache_key = hashlib.sha256(
            json.dumps(
                [messages, want_json],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        started = self.clock()

        # Never cache requests containing profile FACTS.
        if not personal:
            cached = self.cache.get(cache_key)

            if cached:
                cached_at, cached_result = cached

                if started - cached_at < self.CACHE_TTL_S:
                    self.last = {
                        "provider": "cache",
                        "model": None,
                    }

                    return cached_result

                del self.cache[cache_key]

        result, config = self._request_cloud(
            messages,
            kw,
            want_json=want_json,
        )

        if result is None:
            raise LLMUnavailable(
                "no cloud model answered"
            )

        if not personal:
            self.cache[cache_key] = (
                started,
                result,
            )

            while len(self.cache) > self.CACHE_MAX:
                self.cache.popitem(last=False)

        return result


_REDACT = [
    (
        _re.compile(
            r"(avoids loading:\s*)[^;\n]+",
            _re.I,
        ),
        r"\1(not shared)",
    ),
    (
        _re.compile(
            r"(Avoiding exercises that load:\s*)[^.\n]+",
            _re.I,
        ),
        r"\1(not shared)",
    ),
    (
        _re.compile(
            r"(limitations? \()[^)]*(\))",
            _re.I,
        ),
        r"\1not shared\2",
    ),
]


def redact(text):
    """
    Remove stated user limitations from text sent to cloud providers.
    """

    for pattern, replacement in _REDACT:
        text = pattern.sub(replacement, text)

    return text
