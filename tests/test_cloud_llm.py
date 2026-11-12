from agent.cloud_llm import CloudOnlyLLM, LLMUnavailable
from agent.model_registry import MODEL_REGISTRY


def _reply(text="grounded response [S1]"):
    return {"choices": [{"message": {"content": text}}], "usage": {}}


def test_groq_qwen_is_primary():
    calls = []

    def post(url, key, body, timeout):
        calls.append((url, body, timeout))
        return _reply()

    llm = CloudOnlyLLM(post=post, env={
        "GROQ_API_KEY": "groq-test",
        "OPENROUTER_API_KEY": "openrouter-test",
    })

    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Explain protein"}]
    )

    primary = MODEL_REGISTRY[0]

    assert result["choices"][0]["message"]["content"] == "grounded response [S1]"
    assert calls[0][0] == primary["url"]
    assert calls[0][1]["model"] == primary["model"]
    assert llm.last["provider"] == primary["provider"]
    assert len(calls) == 1


def test_openrouter_is_fallback_when_primary_groq_model_fails():
    calls = []

    def post(url, key, body, timeout):
        calls.append((url, body))
        if body["model"] == MODEL_REGISTRY[0]["model"]:
            raise RuntimeError("simulated model failure")
        return _reply()

    llm = CloudOnlyLLM(post=post, env={
        "GROQ_API_KEY": "groq-test",
        "OPENROUTER_API_KEY": "openrouter-test",
    })

    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Explain protein"}]
    )

    fallback = MODEL_REGISTRY[1]

    assert len(calls) == 2
    assert calls[0][1]["model"] == MODEL_REGISTRY[0]["model"]
    assert calls[1][1]["model"] == fallback["model"]
    assert llm.last["provider"] == fallback["provider"]


def test_default_deadline_allows_all_three_model_attempts():
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    calls = []

    def post(url, key, body, timeout):
        calls.append((body["model"], timeout))
        clock.now += timeout
        raise TimeoutError("simulated provider timeout")

    llm = CloudOnlyLLM(
        post=post,
        env={
            "GROQ_API_KEY": "groq-test",
            "OPENROUTER_API_KEY": "openrouter-test",
        },
        clock=clock,
    )

    try:
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "Explain protein"}]
        )
    except LLMUnavailable:
        pass
    else:
        raise AssertionError("expected LLMUnavailable when all models fail")

    assert [model for model, _ in calls] == [
        config["model"] for config in MODEL_REGISTRY
    ]
    assert all(timeout <= 14.0 for _, timeout in calls)
    assert sum(timeout for _, timeout in calls) == 18.0

def test_provider_failure_is_logged_without_secrets_or_prompt(caplog):
    def post(url, key, body, timeout):
        if body["model"] == MODEL_REGISTRY[0]["model"]:
            raise RuntimeError("simulated provider failure")
        return _reply()

    llm = CloudOnlyLLM(post=post, env={
        "GROQ_API_KEY": "secret-groq-test",
        "OPENROUTER_API_KEY": "secret-openrouter-test",
    })

    llm.create_chat_completion(
        messages=[{"role": "user", "content": "private test prompt"}]
    )

    log = caplog.text
    assert "Cloud LLM attempt failed" in log
    assert MODEL_REGISTRY[0]["provider"] in log
    assert MODEL_REGISTRY[0]["model"] in log
    assert "RuntimeError" in log
    assert "secret-groq-test" not in log
    assert "private test prompt" not in log
