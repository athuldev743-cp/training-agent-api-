from agent.cloud_llm import CloudOnlyLLM, LLMUnavailable, PROVIDERS


def _reply(text="grounded response [S1]"):
    return {"choices": [{"message": {"content": text}}], "usage": {}}


def test_groq_is_primary_and_gemini_flash_is_first_fallback():
    calls = []

    def post(url, key, body, timeout):
        calls.append((url, body, timeout))
        return _reply()

    llm = CloudOnlyLLM(post=post, env={
        "GROQ_API_KEY": "groq-test",
        "GEMINI_API_KEY": "gemini-test",
        "OPENROUTER_API_KEY": "openrouter-test",
    })

    result = llm.create_chat_completion(messages=[{"role": "user", "content": "Explain protein"}])

    assert result["choices"][0]["message"]["content"] == "grounded response [S1]"
    assert calls[0][0] == PROVIDERS[0][1]
    assert calls[0][1]["model"] == "openai/gpt-oss-120b"
    assert llm.last["provider"] == "groq"
    assert len(calls) == 1


def test_gemini_flash_is_used_when_groq_fails():
    calls = []

    def post(url, key, body, timeout):
        calls.append((url, body))
        if url == PROVIDERS[0][1]:
            raise RuntimeError("simulated provider failure")
        return _reply()

    llm = CloudOnlyLLM(post=post, env={
        "GROQ_API_KEY": "groq-test",
        "GEMINI_API_KEY": "gemini-test",
        "OPENROUTER_API_KEY": "openrouter-test",
    })

    llm.create_chat_completion(messages=[{"role": "user", "content": "Explain protein"}])

    assert [url for url, _ in calls] == [PROVIDERS[0][1], PROVIDERS[1][1]]
    assert calls[1][1]["model"] == "gemini-2.5-flash"
    assert llm.last["provider"] == "gemini"


def test_default_deadline_allows_all_three_provider_attempts():
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    calls = []

    def post(url, key, body, timeout):
        calls.append((url, timeout))
        clock.now += timeout
        raise TimeoutError("simulated provider timeout")

    llm = CloudOnlyLLM(post=post, env={
        "GROQ_API_KEY": "groq-test",
        "GEMINI_API_KEY": "gemini-test",
        "OPENROUTER_API_KEY": "openrouter-test",
    }, clock=clock)

    try:
        llm.create_chat_completion(messages=[{"role": "user", "content": "Explain protein"}])
    except LLMUnavailable:
        pass
    else:
        raise AssertionError("expected LLMUnavailable when all providers fail")

    assert [item[0] for item in calls] == [provider[1] for provider in PROVIDERS]
    assert all(timeout <= 8.0 for _, timeout in calls)
    assert sum(timeout for _, timeout in calls) == 24.0
