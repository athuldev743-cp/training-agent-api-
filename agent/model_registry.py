
"""
Ordered cloud model registry.

The order determines model priority.
API keys remain in environment variables.
Do not put credentials in this file.
"""

MODEL_REGISTRY = (
    {
        "name": "groq-qwen-3.8-27b",
        "provider": "groq",
        "model": "qwen/qwen3.8-27b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "timeout_s": 5.0,
    },
    {
        "name": "openrouter-nemotron-3-ultra",
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "timeout_s": 14.0,
    },
    {
        "name": "groq-gpt-oss-20b",
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "timeout_s": 5.0,
    },
)
