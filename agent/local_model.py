"""Thin wrapper around the local model. The rest of the project only uses this file."""
import os
import re
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3-1.7B-Q4_K_M.gguf"


def load_model(n_ctx: int = 2048):
    from llama_cpp import Llama  # imported here so the hosted API does not need llama-cpp-python
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=n_ctx,
        n_threads=os.cpu_count(),
        verbose=False,
    )


def strip_think(text: str) -> str:
    """Remove a <think>...</think> block if the model wrote one."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
