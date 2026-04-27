from __future__ import annotations
import asyncio
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
SMALL_MODEL = "qwen3:8b"
LARGE_MODEL = "qwen3:32b"

_MODEL_LOCK = asyncio.Lock()


async def _generate(model: str, prompt: str, keep_alive: str | int) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


async def _unload(model: str) -> None:
    payload = {"model": model, "prompt": "", "keep_alive": 0}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()


async def query_ollama(model: str, prompt: str, keep_alive: str | int = "5m") -> str:
    async with _MODEL_LOCK:
        return await _generate(model, prompt, keep_alive)


async def unload_model(model: str) -> None:
    async with _MODEL_LOCK:
        await _unload(model)


async def query_with_swap(
    small_model: str, large_model: str, prompt_small: str, prompt_large: str
) -> tuple[str, str]:
    async with _MODEL_LOCK:
        small_result = await _generate(small_model, prompt_small, keep_alive=0)
        await _unload(small_model)
        large_result = await _generate(large_model, prompt_large, keep_alive="5m")
        return small_result, large_result
