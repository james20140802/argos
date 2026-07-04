from __future__ import annotations

import httpx
import pytest

from argos.brain import ollama_client
from argos.brain.ollama_client import OllamaInfraError


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.HTTPStatusError("500", request=httpx.Request("POST", "http://x"), response=httpx.Response(500)),
    ],
)
async def test_query_ollama_wraps_infra_failures(monkeypatch, exc):
    async def _boom(*a, **k):
        raise exc

    monkeypatch.setattr(ollama_client, "_generate", _boom)
    with pytest.raises(OllamaInfraError):
        await ollama_client.query_ollama("qwen3:8b", "hi")


@pytest.mark.asyncio
async def test_query_ollama_passes_through_non_infra(monkeypatch):
    async def _boom(*a, **k):
        raise ValueError("not infra")

    monkeypatch.setattr(ollama_client, "_generate", _boom)
    with pytest.raises(ValueError):
        await ollama_client.query_ollama("qwen3:8b", "hi")


@pytest.mark.asyncio
async def test_query_ollama_passes_through_4xx(monkeypatch):
    err = httpx.HTTPStatusError(
        "400", request=httpx.Request("POST", "http://x"), response=httpx.Response(400)
    )

    async def _boom(*a, **k):
        raise err

    monkeypatch.setattr(ollama_client, "_generate", _boom)
    with pytest.raises(httpx.HTTPStatusError):
        await ollama_client.query_ollama("qwen3:8b", "hi")
