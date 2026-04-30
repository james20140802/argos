from __future__ import annotations
import asyncio
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
SMALL_MODEL = "qwen3:8b"
LARGE_MODEL = "qwen3:32b"

DEFAULT_TIMEOUT = 120
LARGE_MODEL_TIMEOUT = 600
DEFAULT_NUM_CTX = 4096

_MODEL_LOCK = asyncio.Lock()


async def _generate(
    model: str,
    prompt: str,
    keep_alive: str | int,
    timeout: float = DEFAULT_TIMEOUT,
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool | None = None,
) -> str:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_ctx": num_ctx},
    }
    if think is not None:
        payload["think"] = think
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10)) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


async def _unload(model: str) -> None:
    payload = {"model": model, "prompt": "", "keep_alive": 0}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()


async def query_ollama(
    model: str,
    prompt: str,
    keep_alive: str | int = "5m",
    timeout: float = DEFAULT_TIMEOUT,
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool | None = None,
) -> str:
    async with _MODEL_LOCK:
        return await _generate(model, prompt, keep_alive, timeout, num_ctx, think)


async def unload_model(model: str) -> None:
    async with _MODEL_LOCK:
        await _unload(model)


async def prewarm_model(
    model: str,
    keep_alive: str | int = "5m",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> None:
    """Force Ollama to load `model` into memory so a later inference call avoids the cold-load.

    `num_ctx` must match the value the real call will use; otherwise Ollama re-loads the
    model with a different context size and the prewarm is wasted.
    """
    async with _MODEL_LOCK:
        await _generate(
            model, "", keep_alive, timeout=LARGE_MODEL_TIMEOUT, num_ctx=num_ctx
        )


async def query_with_swap(
    small_model: str, large_model: str, prompt_small: str, prompt_large: str
) -> tuple[str, str]:
    async with _MODEL_LOCK:
        small_result = await _generate(small_model, prompt_small, keep_alive=0)
        await _unload(small_model)
        large_result = await _generate(large_model, prompt_large, keep_alive="5m")
        return small_result, large_result
