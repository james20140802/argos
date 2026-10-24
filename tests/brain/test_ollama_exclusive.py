"""ARG-310: query_exclusive -- unload other resident models, then generate."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from argos.brain.ollama_client import OLLAMA_BASE_URL, OllamaInfraError, query_exclusive


def _bodies(route) -> list[dict]:
    return [json.loads(c.request.content) for c in route.calls]


def _ps(models: list[str]):
    return respx.get(f"{OLLAMA_BASE_URL}/api/ps").mock(
        return_value=httpx.Response(200, json={"models": [{"name": m} for m in models]})
    )


@pytest.mark.asyncio
async def test_unloads_other_models_then_generates():
    with respx.mock:
        _ps(["qwen3:8b", "gemma4:26b"])
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        assert await query_exclusive("gemma4:26b", "p", num_ctx=16384) == "ok"
        bodies = _bodies(route)
        assert [(b["model"], b["keep_alive"]) for b in bodies] == [
            ("qwen3:8b", 0),
            ("gemma4:26b", "5m"),
            ("gemma4:26b", 0),  # unload_after: released before the lock is
        ]
        assert bodies[1]["options"]["num_ctx"] == 16384


@pytest.mark.asyncio
async def test_no_unload_when_nothing_else_loaded():
    with respx.mock:
        _ps([])
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        await query_exclusive("gemma4:26b", "p")
        assert len(route.calls) == 2  # generate + post-use unload
        await query_exclusive("gemma4:26b", "p", unload_after=False)
        assert len(route.calls) == 3


@pytest.mark.asyncio
async def test_ps_failure_fails_closed_without_generating():
    with respx.mock:
        respx.get(f"{OLLAMA_BASE_URL}/api/ps").mock(return_value=httpx.Response(500))
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        with pytest.raises(OllamaInfraError):
            await query_exclusive("gemma4:26b", "p")
        assert len(route.calls) == 0


@pytest.mark.asyncio
async def test_unload_failure_fails_closed_without_generating():
    with respx.mock:
        _ps(["qwen3:8b"])
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(OllamaInfraError):
            await query_exclusive("gemma4:26b", "p")
        assert [b["model"] for b in _bodies(route)] == ["qwen3:8b"]  # no gemma generate


@pytest.mark.asyncio
async def test_unloads_model_before_releasing_lock(monkeypatch):
    """A queued Deep Dive-style query must only start after the summary model is gone."""
    import argos.brain.ollama_client as oc

    events: list[str] = []

    async def fake_ps():
        return []

    async def fake_unload(model):
        events.append(f"unload:{model}")

    async def fake_raw(model, *a, **k):
        events.append(f"generate:{model}")
        return {"response": "x"}

    async def fake_gen(model, *a, **k):
        events.append(f"other:{model}")
        return "y"

    monkeypatch.setattr(oc, "_MODEL_LOCK", asyncio.Lock())
    monkeypatch.setattr(oc, "loaded_models", fake_ps)
    monkeypatch.setattr(oc, "_unload", fake_unload)
    monkeypatch.setattr(oc, "_generate_raw", fake_raw)
    monkeypatch.setattr(oc, "_generate", fake_gen)

    t1 = asyncio.create_task(oc.query_exclusive("gemma4:26b", "p"))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(oc.query_ollama("large32", "q"))
    await asyncio.gather(t1, t2)
    assert events == ["generate:gemma4:26b", "unload:gemma4:26b", "other:large32"]


@pytest.mark.asyncio
async def test_fills_stats_out():
    with respx.mock:
        _ps([])
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "response": "ok",
                    "prompt_eval_count": 123,
                    "eval_count": 45,
                    "total_duration": 6_000_000_000,
                },
            )
        )
        stats: dict = {}
        await query_exclusive("gemma4:26b", "p", stats_out=stats)
        assert stats == {
            "prompt_eval_count": 123,
            "eval_count": 45,
            "total_duration": 6_000_000_000,
        }


@pytest.mark.asyncio
async def test_5xx_maps_to_infra_error():
    with respx.mock:
        _ps([])
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(return_value=httpx.Response(500))
        with pytest.raises(OllamaInfraError):
            await query_exclusive("gemma4:26b", "p")


@pytest.mark.asyncio
async def test_transport_error_maps_to_infra_error():
    with respx.mock:
        _ps([])
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            side_effect=httpx.ConnectError("down")
        )
        with pytest.raises(OllamaInfraError):
            await query_exclusive("gemma4:26b", "p")


@pytest.mark.asyncio
async def test_holds_lock_across_unload_and_generate(monkeypatch):
    """Another query_ollama must not run between the unload and the generate."""
    import argos.brain.ollama_client as oc

    events: list[str] = []
    in_unload = asyncio.Event()
    release = asyncio.Event()

    async def fake_ps():
        return ["qwen3:8b"]

    async def fake_unload(model):
        events.append(f"unload:{model}")
        in_unload.set()
        await release.wait()

    async def fake_raw(model, *a, **k):
        events.append(f"generate:{model}")
        return {"response": "x"}

    # Fresh lock: contention binds an asyncio.Lock to this test's event loop.
    monkeypatch.setattr(oc, "_MODEL_LOCK", asyncio.Lock())
    monkeypatch.setattr(oc, "loaded_models", fake_ps)
    monkeypatch.setattr(oc, "_unload", fake_unload)
    monkeypatch.setattr(oc, "_generate_raw", fake_raw)

    t1 = asyncio.create_task(oc.query_exclusive("gemma4:26b", "p"))
    await in_unload.wait()
    t2 = asyncio.create_task(oc.query_ollama("qwen3:8b", "other"))
    await asyncio.sleep(0.05)
    assert events == ["unload:qwen3:8b"]  # t2 is still blocked on the lock
    release.set()
    await asyncio.gather(t1, t2)
    assert events == [
        "unload:qwen3:8b",
        "generate:gemma4:26b",
        "unload:gemma4:26b",
        "generate:qwen3:8b",
    ]


# --- llm_client wiring ---


@pytest.mark.asyncio
async def test_event_summary_client_resolves_configured_model_and_routes_exclusive(monkeypatch):
    from unittest.mock import AsyncMock

    import argos.brain.llm_client as lc
    from argos.config import UserConfig

    class _S:
        user = UserConfig(event_summary={"model": "custom:7b"})

    monkeypatch.setattr(lc, "_get_settings", lambda: _S)
    excl = AsyncMock(return_value="r")
    plain = AsyncMock(return_value="p")
    monkeypatch.setattr(lc, "query_exclusive", excl)
    monkeypatch.setattr(lc, "query_ollama", plain)
    client = lc.get_event_summary_llm_client()
    assert await client.query("large", "x", exclusive=True, num_ctx=16384) == "r"
    assert excl.await_args.args[0] == "custom:7b"
    assert excl.await_args.kwargs["num_ctx"] == 16384
    assert await client.query("large", "x") == "p"
    assert plain.await_args.args[0] == "custom:7b"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["ps", "unload"])
async def test_event_naming_keeps_fallback_when_cleanup_fails(failure):
    """Cleanup failure -> no generate; naming returns None (caller keeps fallback + stale)."""
    from argos.brain.event_naming import EvidenceDoc, generate_event_naming

    with respx.mock:
        if failure == "ps":
            respx.get(f"{OLLAMA_BASE_URL}/api/ps").mock(return_value=httpx.Response(500))
        else:
            _ps(["qwen3:8b"])
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(500)
        )
        result = await generate_event_naming([EvidenceDoc(title="a", summary="b")])
        assert result is None
        assert all(b["model"] != "gemma4:26b" or b["keep_alive"] == 0 for b in _bodies(route))
        assert not any(b.get("prompt") for b in _bodies(route))  # nothing was generated


@pytest.mark.asyncio
async def test_unloads_model_even_when_generation_times_out(monkeypatch):
    import argos.brain.ollama_client as oc

    unloaded: list[str] = []

    async def fake_ps():
        return []

    async def fake_unload(model):
        unloaded.append(model)

    async def fake_raw(*args, **kwargs):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(oc, "_MODEL_LOCK", asyncio.Lock())
    monkeypatch.setattr(oc, "loaded_models", fake_ps)
    monkeypatch.setattr(oc, "_unload", fake_unload)
    monkeypatch.setattr(oc, "_generate_raw", fake_raw)
    with pytest.raises(OllamaInfraError):
        await query_exclusive("gemma4:26b", "p")
    assert unloaded == ["gemma4:26b"]
    assert not oc._MODEL_LOCK.locked()
