from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

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


def _get_settings() -> Any:
    """Return the global settings singleton.

    Extracted as a standalone function so tests can monkeypatch it without
    needing to reach into the module-level ``settings`` import.
    """
    from argos.config import settings

    return settings


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

    async def prewarm(
        self,
        model_role: Literal["small", "large"],
        num_ctx: int | None = None,
    ) -> None: ...

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
    def __init__(self, settings: Any = None) -> None:
        if settings is None:
            settings = _get_settings()
        self._small = settings.user.ollama.model_triage
        # The genealogist model is the authoritative large-model config.
        # `ollama.model_deepdive` is preserved for legacy references but the
        # genealogist node drives the actual 32B choice via genealogist.model.
        self._large = settings.user.genealogist.model
        self._genealogist_num_ctx = settings.user.genealogist.num_ctx
        # Host is propagated via ollama_client._base_url(), which reads
        # settings.user.ollama.host at call time, so all inference respects
        # the configured host rather than the hard-coded default.

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

    async def prewarm(
        self,
        model_role: Literal["small", "large"],
        num_ctx: int | None = None,
    ) -> None:
        """Prewarm ``model_role`` in Ollama, allocating the correct KV cache size.

        When ``num_ctx`` is ``None`` the value is taken from
        ``genealogist.num_ctx`` for the large role (so prewarm and the
        subsequent inference call both allocate the same context size — if they
        differ Ollama reloads the model and the prewarm is wasted).  For the
        small role it falls back to ``DEFAULT_NUM_CTX``.
        """
        resolved_num_ctx: int
        if num_ctx is not None:
            resolved_num_ctx = num_ctx
        elif model_role == "large":
            resolved_num_ctx = self._genealogist_num_ctx
        else:
            resolved_num_ctx = DEFAULT_NUM_CTX
        await prewarm_model(self._resolve(model_role), num_ctx=resolved_num_ctx)

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
    settings = _get_settings()
    backend = settings.user.llm.backend
    if backend == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
