"""Cloud-first chat with local fallback. Free providers in order, each tried once; stops at the first valid reply.
Failing providers are skipped for a cooldown, and the whole chain has a total deadline. Router calls, streaming and
prompts containing profile FACTS always stay local. Off unless CLOUD_LLM=1."""
import json, os, time, urllib.request

PROVIDERS = [
    ("groq", "https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
]


def _post(url, key, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "local-training-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class HybridLLM:
    COOLDOWN_S = 300

    def __init__(self, local, post=_post, timeout=8.0, env=None, deadline=10.0, clock=time.monotonic):
        if env is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except Exception:
                pass
            env = os.environ
        self.local, self._post, self.timeout, self.env = local, post, timeout, env
        self.deadline, self.clock, self.skip_until = deadline, clock, {}
        self.last, self.errors = {"provider": None}, []

    def _cloud_ok(self, kw):
        if self.env.get("CLOUD_LLM") != "1" or kw.get("stream") or kw.get("response_format"):
            return False
        return not any("FACTS:" in str(m.get("content", "")) for m in kw.get("messages", []))

    def _fail(self, name, why):
        self.errors.append((name, why))  # type only: messages can contain request details
        self.skip_until[name] = self.clock() + self.COOLDOWN_S

    def create_chat_completion(self, **kw):
        if self._cloud_ok(kw):
            start = self.clock()
            msgs = [{**m, "content": str(m.get("content", "")).replace(" /no_think", "")} for m in kw["messages"]]
            for name, url, keyvar, modelvar, default in PROVIDERS:
                key = self.env.get(keyvar)
                if not key:
                    continue
                now = self.clock()
                if now < self.skip_until.get(name, 0):
                    continue
                remaining = self.deadline - (now - start)
                if remaining <= 0.5:
                    break
                body = {"model": self.env.get(modelvar) or default, "messages": msgs,
                        "max_tokens": min(800, max(400, 4 * kw.get("max_tokens", 160))), "temperature": kw.get("temperature", 0.2)}
                try:
                    data = self._post(url, key, body, min(self.timeout, remaining))
                    choice = data["choices"][0]
                    text = choice["message"]["content"]
                    if choice.get("finish_reason") == "length":
                        self._fail(name, "truncated")
                        continue
                    if isinstance(text, str) and text.strip():
                        self.last = {"provider": name}
                        return {"choices": [{"message": {"content": text.strip()}}], "usage": data.get("usage") or {}}
                    self._fail(name, "empty_response")
                except Exception as e:
                    self._fail(name, type(e).__name__)
        self.last = {"provider": "local"}
        return self.local.create_chat_completion(**kw)


import collections
import hashlib


class LLMUnavailable(RuntimeError):
    """No cloud provider produced a valid reply."""


class CloudOnlyLLM(HybridLLM):
    """No local model. Providers in order, each tried once; raises LLMUnavailable if none answers.
    Structured (JSON) requests must come back as valid JSON. Identical requests that carry no personal facts are cached."""
    CACHE_TTL_S, CACHE_MAX = 6 * 3600, 500

    def __init__(self, post=_post, timeout=8.0, env=None, deadline=10.0, clock=time.monotonic):
        super().__init__(None, post=post, timeout=timeout, env=env, deadline=deadline, clock=clock)
        self.cache = collections.OrderedDict()

    def create_chat_completion(self, **kw):
        if kw.get("stream"):
            raise LLMUnavailable("streaming is not supported by the cloud chain")
        msgs = [{**m, "content": str(m.get("content", "")).replace(" /no_think", "")} for m in kw["messages"]]
        if self.env.get("CLOUD_SHARE_LIMITATIONS") != "1":
            msgs = [{**m, "content": redact(m["content"])} for m in msgs]
        personal = any("FACTS:" in m["content"] for m in msgs)
        want_json = bool(kw.get("response_format"))
        key = hashlib.sha256(json.dumps([msgs, want_json], sort_keys=True).encode("utf-8")).hexdigest()
        start = self.clock()
        if not personal:
            hit = self.cache.get(key)
            if hit and start - hit[0] < self.CACHE_TTL_S:
                self.last = {"provider": "cache"}
                return hit[1]
        for name, url, keyvar, modelvar, default in PROVIDERS:
            api_key = self.env.get(keyvar)
            if not api_key:
                continue
            now = self.clock()
            if now < self.skip_until.get(name, 0):
                continue
            remaining = self.deadline - (now - start)
            if remaining <= 0.5:
                break
            body = {"model": self.env.get(modelvar) or default, "messages": msgs,
                    "max_tokens": min(800, max(400, 4 * kw.get("max_tokens", 160))), "temperature": kw.get("temperature", 0.2)}
            if want_json:
                body["response_format"] = {"type": "json_object"}
            try:
                data = self._post(url, api_key, body, min(self.timeout, remaining))
                choice = data["choices"][0]
                text = choice["message"]["content"]
                if choice.get("finish_reason") == "length":
                    self._fail(name, "truncated"); continue
                if not (isinstance(text, str) and text.strip()):
                    self._fail(name, "empty_response"); continue
                if want_json:
                    json.loads(text)
                result = {"choices": [{"message": {"content": text.strip()}}], "usage": data.get("usage") or {}}
                self.last = {"provider": name}
                if not personal:
                    self.cache[key] = (start, result)
                    while len(self.cache) > self.CACHE_MAX:
                        self.cache.popitem(last=False)
                return result
            except json.JSONDecodeError:
                self._fail(name, "bad_json")
            except Exception as e:
                self._fail(name, type(e).__name__)
        raise LLMUnavailable("no cloud provider answered")


import re as _re

_REDACT = [
    (_re.compile(r"(avoids loading:\s*)[^;\n]+", _re.I), r"\1(not shared)"),
    (_re.compile(r"(Avoiding exercises that load:\s*)[^.\n]+", _re.I), r"\1(not shared)"),
    (_re.compile(r"(limitations? \()[^)]*(\))", _re.I), r"\1not shared\2"),
]


def redact(text):
    """Remove the user's stated limitations (health information) from text bound for a free-tier cloud provider."""
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text
