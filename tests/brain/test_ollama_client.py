from __future__ import annotations
import asyncio
import pytest
import respx
import httpx
from argos.brain.ollama_client import (
    LARGE_MODEL_TIMEOUT,
    OLLAMA_BASE_URL,
    _MODEL_LOCK,
    prewarm_model,
    query_ollama,
    unload_model,
)

@pytest.mark.asyncio
async def test_query_ollama_returns_response():
    import json

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "hello"})
        )
        result = await query_ollama("qwen3:8b", "test prompt")
        assert result == "hello"
        body = json.loads(route.calls[0].request.content)
        # The default num_ctx must be sent so Ollama does not auto-allocate a 32K KV cache
        # that spills layers to CPU on a 32GB Mac (see fix/genealogist-ollama-timeout-and-prewarm).
        assert body["options"]["num_ctx"] == 4096

@pytest.mark.asyncio
async def test_unload_model_sends_keep_alive_zero():
    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        await unload_model("qwen3:8b")
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
    valid_sequences = [
        ["start-a", "end-a", "start-b", "end-b"],
        ["start-b", "end-b", "start-a", "end-a"],
    ]
    assert call_order in valid_sequences, f"Lock did not serialize calls, got: {call_order}"


@pytest.mark.asyncio
async def test_query_ollama_raises_on_http_error():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(500, json={"error": "internal server error"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            from argos.brain.ollama_client import query_ollama
            await query_ollama("qwen3:8b", "test prompt")


@pytest.mark.asyncio
async def test_query_ollama_accepts_explicit_timeout():
    import json

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        result = await query_ollama("qwen3:32b", "p", keep_alive="5m", timeout=600)

    assert result == "ok"
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:32b"
    assert body["keep_alive"] == "5m"


@pytest.mark.asyncio
async def test_query_ollama_propagates_num_ctx():
    import json

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        await query_ollama("qwen3:32b", "p", num_ctx=8192)

    body = json.loads(route.calls[0].request.content)
    assert body["options"]["num_ctx"] == 8192


@pytest.mark.asyncio
async def test_query_ollama_propagates_think_when_set():
    import json

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        await query_ollama("qwen3:32b", "p", think=False)

    body = json.loads(route.calls[0].request.content)
    assert body["think"] is False


@pytest.mark.asyncio
async def test_query_ollama_omits_think_by_default():
    import json

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        await query_ollama("qwen3:8b", "p")

    body = json.loads(route.calls[0].request.content)
    assert "think" not in body


@pytest.mark.asyncio
async def test_prewarm_model_sends_empty_prompt_with_keep_alive():
    import json

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        await prewarm_model("qwen3:32b")

    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:32b"
    assert body["prompt"] == ""
    assert body["keep_alive"] == "5m"
    # Prewarm must allocate the same KV cache the real call will use, otherwise Ollama
    # reloads with a different context size and the prewarm is wasted.
    assert body["options"]["num_ctx"] == 4096


@pytest.mark.asyncio
async def test_large_model_timeout_constant_is_generous():
    # Sanity: the qwen3:32b cold-load + generate budget needs to clearly exceed
    # the 120s default that was triggering ReadTimeout warnings.
    assert LARGE_MODEL_TIMEOUT >= 300


@pytest.mark.asyncio
async def test_query_with_swap_calls_both_models():
    import json
    from argos.brain.ollama_client import query_with_swap

    responses = {"qwen3:8b": "small-answer", "qwen3:32b": "large-answer"}

    def _handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"response": responses.get(body["model"], "")})

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(side_effect=_handler)
        small, large = await query_with_swap("qwen3:8b", "qwen3:32b", "small-prompt", "large-prompt")

    assert small == "small-answer"
    assert large == "large-answer"
