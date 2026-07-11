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


class OllamaInfraError(Exception):
    """Raised when a call to Ollama fails for infrastructure reasons.

    Covers connection failures, timeouts, and server errors (HTTP 5xx —
    Ollama returns 500 on VRAM OOM). Distinct from a legitimate model
    response that fails parsing/validation downstream (ARG-190/ARG-214):
    those are not infra failures and must not raise this type.
    """


def _base_url() -> str:
    """Return the Ollama base URL from config, falling back to the default."""
    try:
        from argos.config import settings

        host = settings.user.ollama.host
        if host:
            return host.rstrip("/")
    except Exception:
        pass
    return OLLAMA_BASE_URL


async def _generate(
    model: str,
    prompt: str,
    keep_alive: str | int,
    timeout: float = DEFAULT_TIMEOUT,
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool | None = None,
    temperature: float | None = None,
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
    if temperature is not None:
        # ARG-206: triage now extracts a deterministic rubric — temperature=0
        # keeps the 5-field JSON extraction as reproducible as an LLM call
        # can be, same mechanism as num_ctx (an Ollama "options" entry).
        payload["options"]["temperature"] = temperature
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10)) as client:
        resp = await client.post(f"{_base_url()}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


async def _unload(model: str) -> None:
    payload = {"model": model, "prompt": "", "keep_alive": 0}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{_base_url()}/api/generate", json=payload)
        resp.raise_for_status()


async def query_ollama(
    model: str,
    prompt: str,
    keep_alive: str | int = "5m",
    timeout: float = DEFAULT_TIMEOUT,
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool | None = None,
    temperature: float | None = None,
) -> str:
    async with _MODEL_LOCK:
        try:
            return await _generate(model, prompt, keep_alive, timeout, num_ctx, think, temperature)
        except httpx.HTTPStatusError as exc:
            # raise_for_status() failure: 5xx (incl. Ollama's 500 on VRAM OOM)
            # is infra; 4xx is a real request bug and must surface untouched.
            if exc.response.status_code >= 500:
                raise OllamaInfraError(str(exc)) from exc
            raise
        except httpx.TransportError as exc:
            # Covers ConnectError/ConnectTimeout, every read/write timeout, and
            # dropped-connection errors (ReadError, WriteError, RemoteProtocolError)
            # Ollama raises when it resets the socket mid-request under OOM/crash.
            # All are infra failures the pipeline must retain+retry, not drop. (ARG-190)
            raise OllamaInfraError(str(exc)) from exc


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


async def batch_embed(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """Embed multiple texts in a single HTTP round-trip via /api/embed."""
    payload = {"model": model, "input": texts, "keep_alive": 0}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
        resp = await client.post(f"{_base_url()}/api/embed", json=payload)
        resp.raise_for_status()
        return resp.json()["embeddings"]


async def embed(text: str, model: str = "nomic-embed-text") -> list[float]:
    """Embed a single text string and return the 768-dim vector."""
    results = await batch_embed([text], model=model)
    return results[0]


async def unload_then_query(
    unload: str,
    model: str,
    prompt: str,
    keep_alive: str | int = "5m",
    timeout: float = DEFAULT_TIMEOUT,
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool | None = None,
    temperature: float | None = None,
) -> str:
    """Unload `unload` and run a query on `model` atomically under a single lock.

    Holds `_MODEL_LOCK` across both calls so a concurrent small-model query
    cannot slip in between unload and load — preserving the one-model-at-a-time
    VRAM invariant.
    """
    async with _MODEL_LOCK:
        await _unload(unload)
        return await _generate(model, prompt, keep_alive, timeout, num_ctx, think, temperature)
