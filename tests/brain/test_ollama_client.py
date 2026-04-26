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


@pytest.mark.asyncio
async def test_query_ollama_raises_on_http_error():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(500, json={"error": "internal server error"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            from argos.brain.ollama_client import query_ollama
            await query_ollama("llama3:8b", "test prompt")


@pytest.mark.asyncio
async def test_query_with_swap_calls_both_models():
    import json
    from argos.brain.ollama_client import query_with_swap

    responses = {"llama3:8b": "small-answer", "llama3:70b": "large-answer"}

    def _handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"response": responses.get(body["model"], "")})

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(side_effect=_handler)
        small, large = await query_with_swap("llama3:8b", "llama3:70b", "small-prompt", "large-prompt")

    assert small == "small-answer"
    assert large == "large-answer"
