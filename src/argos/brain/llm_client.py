from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from argos.brain.ollama_client import (
    DEFAULT_NUM_CTX,
    LARGE_MODEL_TIMEOUT,
    prewarm_model,
    query_ollama,
    unload_model,
    unload_then_query,
)

if TYPE_CHECKING:
    pass

_DEFAULT_SMALL_TIMEOUT = 120


@runtime_checkable
class LLMClient(Protocol):
    async def query(
        self,
        model_role: Literal["small", "large"],
        prompt: str,
        keep_alive: str | int = "5m",
        timeout: float | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        think: bool | None = None,
    ) -> str: ...

    async def unload(self, model_role: Literal["small", "large"]) -> None: ...

    async def prewarm(self, model_role: Literal["small", "large"]) -> None: ...

    async def unload_then_query(
        self,
        unload_role: Literal["small", "large"],
        query_role: Literal["small", "large"],
        prompt: str,
        keep_alive: str | int = "5m",
        timeout: float | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        think: bool | None = None,
    ) -> str: ...


class OllamaClient:
    def __init__(self) -> None:
        from argos.config import settings

        self._small = settings.user.ollama.model_triage
        self._large = settings.user.ollama.model_deepdive

    def _resolve(self, role: Literal["small", "large"]) -> str:
        return self._small if role == "small" else self._large

    def _default_timeout(self, role: Literal["small", "large"]) -> float:
        return LARGE_MODEL_TIMEOUT if role == "large" else _DEFAULT_SMALL_TIMEOUT

    async def query(
        self,
        model_role: Literal["small", "large"],
        prompt: str,
        keep_alive: str | int = "5m",
        timeout: float | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        think: bool | None = None,
    ) -> str:
        return await query_ollama(
            self._resolve(model_role),
            prompt,
            keep_alive=keep_alive,
            timeout=timeout if timeout is not None else self._default_timeout(model_role),
            num_ctx=num_ctx,
            think=think,
        )

    async def unload(self, model_role: Literal["small", "large"]) -> None:
        await unload_model(self._resolve(model_role))

    async def prewarm(self, model_role: Literal["small", "large"]) -> None:
        await prewarm_model(self._resolve(model_role))

    async def unload_then_query(
        self,
        unload_role: Literal["small", "large"],
        query_role: Literal["small", "large"],
        prompt: str,
        keep_alive: str | int = "5m",
        timeout: float | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
        think: bool | None = None,
    ) -> str:
        return await unload_then_query(
            self._resolve(unload_role),
            self._resolve(query_role),
            prompt,
            keep_alive=keep_alive,
            timeout=timeout if timeout is not None else self._default_timeout(query_role),
            num_ctx=num_ctx,
            think=think,
        )


def get_llm_client() -> OllamaClient:
    from argos.config import settings

    backend = settings.user.llm.backend
    if backend == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
