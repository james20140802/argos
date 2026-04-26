from __future__ import annotations
import asyncio
import pytest
import respx
import httpx
from argos.brain.ollama_client import query_ollama, unload_model, _MODEL_LOCK, OLLAMA_BASE_URL

@pytest.mark.asyncio
async def test_query_ollama_returns_response():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "hello"})
        )
        result = await query_ollama("llama3:8b", "test prompt")
        assert result == "hello"

@pytest.mark.asyncio
async def test_unload_model_sends_keep_alive_zero():
    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        await unload_model("llama3:8b")
        assert route.called
        sent_payload = route.calls[0].request
        import json
        body = json.loads(sent_payload.content)
        assert body["keep_alive"] == 0

@pytest.mark.asyncio
async def test_model_lock_serializes_calls():
    call_order = []
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        async def _call(tag: str) -> None:
            async with _MODEL_LOCK:
                call_order.append(f"start-{tag}")
                await asyncio.sleep(0.01)
                call_order.append(f"end-{tag}")
        await asyncio.gather(_call("a"), _call("b"))
    assert call_order.index("end-a") < call_order.index("start-b") or \
           call_order.index("end-b") < call_order.index("start-a")
