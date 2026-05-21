"""Tests for argos.brain.pipeline (ARG-39, ARG-54, ARG-87).

Single-URL run_brain_pipeline: 32B prewarm/genealogist skipped on cold start.
Batch run_batch_brain_pipeline: trust-score gate, cold-start, preflight path.
We mock every node so the tests run without Docker or Ollama.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain import pipeline as brain_pipeline
from argos.models.tech_item import CategoryType


def _triaged_state(**overrides):
    base = {
        "raw_text": "x",
        "source_url": "https://example.com",
        "is_valid": True,
        "trust_score": 0.7,
        "summary": "s",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_run_brain_pipeline_skips_prewarm_and_genealogist_on_cold_start(
    monkeypatch,
):
    triaged = _triaged_state()
    cold = _triaged_state(genealogy_skipped=True, genealogy_skip_reason="cold_start")
    saved = {**cold, "saved": True}

    monkeypatch.setattr(
        brain_pipeline, "triage_node", AsyncMock(return_value=triaged)
    )
    monkeypatch.setattr(
        brain_pipeline, "embed_and_search_node", AsyncMock(return_value=cold)
    )
    genealogist_mock = AsyncMock()
    monkeypatch.setattr(brain_pipeline, "genealogist_node", genealogist_mock)
    save_mock = AsyncMock(return_value=saved)
    monkeypatch.setattr(brain_pipeline, "save_node", save_mock)

    prewarm_called = {"n": 0}

    class _FakeClient:
        async def prewarm(self, role):
            prewarm_called["n"] += 1

    monkeypatch.setattr(brain_pipeline, "get_genealogist_llm_client", lambda: _FakeClient())

    session = MagicMock()
    result = await brain_pipeline.run_brain_pipeline("x", "https://e.com", session)

    assert result["genealogy_skipped"] is True
    assert result["saved"] is True
    # Critical assertions for ARG-39:
    genealogist_mock.assert_not_awaited()
    assert prewarm_called["n"] == 0
    save_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_brain_pipeline_runs_prewarm_and_genealogist_when_warm(
    monkeypatch,
):
    triaged = _triaged_state()
    warm = _triaged_state(related_tech_ids=["abc"])
    succession = {**warm, "succession_result": {"relation_type": "Enhance"}}
    saved = {**succession, "saved": True}

    prewarm_started = {"n": 0}
    prewarm_completed = {"n": 0}

    class _FakeClient:
        async def prewarm(self, role):
            prewarm_started["n"] += 1
            # Yield control once so the genealogist coroutine can await us.
            import asyncio as _asyncio
            await _asyncio.sleep(0)
            prewarm_completed["n"] += 1

    genealogist_calls = {"n": 0, "received_prewarm_task": None}

    async def _fake_genealogist(state, *, prewarm_task=None):
        genealogist_calls["n"] += 1
        genealogist_calls["received_prewarm_task"] = prewarm_task
        if prewarm_task is not None:
            await prewarm_task
        return succession

    monkeypatch.setattr(
        brain_pipeline, "triage_node", AsyncMock(return_value=triaged)
    )
    monkeypatch.setattr(
        brain_pipeline, "embed_and_search_node", AsyncMock(return_value=warm)
    )
    monkeypatch.setattr(brain_pipeline, "genealogist_node", _fake_genealogist)
    save_mock = AsyncMock(return_value=saved)
    monkeypatch.setattr(brain_pipeline, "save_node", save_mock)
    monkeypatch.setattr(brain_pipeline, "get_genealogist_llm_client", lambda: _FakeClient())

    session = MagicMock()
    result = await brain_pipeline.run_brain_pipeline("x", "https://e.com", session)

    assert result["saved"] is True
    assert genealogist_calls["n"] == 1
    assert genealogist_calls["received_prewarm_task"] is not None
    assert prewarm_started["n"] == 1
    assert prewarm_completed["n"] == 1
    save_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_brain_pipeline_returns_early_when_triage_fails(monkeypatch):
    rejected = _triaged_state(is_valid=False)
    monkeypatch.setattr(
        brain_pipeline, "triage_node", AsyncMock(return_value=rejected)
    )
    embed_mock = AsyncMock()
    monkeypatch.setattr(brain_pipeline, "embed_and_search_node", embed_mock)
    genealogist_mock = AsyncMock()
    monkeypatch.setattr(brain_pipeline, "genealogist_node", genealogist_mock)
    save_mock = AsyncMock()
    monkeypatch.setattr(brain_pipeline, "save_node", save_mock)

    class _FakeClient:
        async def prewarm(self, role):  # pragma: no cover - must not run
            raise AssertionError("prewarm must not run when triage rejects")

    monkeypatch.setattr(brain_pipeline, "get_genealogist_llm_client", lambda: _FakeClient())

    result = await brain_pipeline.run_brain_pipeline("x", "https://e.com", MagicMock())

    assert result["is_valid"] is False
    embed_mock.assert_not_awaited()
    genealogist_mock.assert_not_awaited()
    save_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# run_brain_pipeline — source_category forwarding (ARG-54)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_brain_pipeline_forwards_source_category_into_initial_state(
    monkeypatch,
):
    """source_category kwarg must be seeded into the initial BrainState."""
    triaged = _triaged_state(source_category=CategoryType.MAINSTREAM)
    cold = _triaged_state(
        genealogy_skipped=True,
        genealogy_skip_reason="cold_start",
        source_category=CategoryType.MAINSTREAM,
    )
    saved = {**cold, "saved": True}

    captured_initial: dict = {}

    async def _fake_triage(state):
        captured_initial.update(state)
        return triaged

    monkeypatch.setattr(brain_pipeline, "triage_node", _fake_triage)
    monkeypatch.setattr(
        brain_pipeline, "embed_and_search_node", AsyncMock(return_value=cold)
    )
    monkeypatch.setattr(brain_pipeline, "genealogist_node", AsyncMock())
    monkeypatch.setattr(
        brain_pipeline, "save_node", AsyncMock(return_value=saved)
    )
    monkeypatch.setattr(
        brain_pipeline, "get_genealogist_llm_client", lambda: MagicMock()
    )

    await brain_pipeline.run_brain_pipeline(
        "x", "https://e.com", MagicMock(), source_category=CategoryType.MAINSTREAM
    )

    assert captured_initial["source_category"] is CategoryType.MAINSTREAM
    assert captured_initial["category"] is None


@pytest.mark.asyncio
async def test_run_brain_pipeline_defaults_source_category_to_none(monkeypatch):
    """When called without source_category, initial state must have None for both fields."""
    captured_initial: dict = {}

    rejected = _triaged_state(is_valid=False)

    async def _fake_triage(state):
        captured_initial.update(state)
        return rejected

    monkeypatch.setattr(brain_pipeline, "triage_node", _fake_triage)
    monkeypatch.setattr(brain_pipeline, "embed_and_search_node", AsyncMock())
    monkeypatch.setattr(brain_pipeline, "genealogist_node", AsyncMock())
    monkeypatch.setattr(brain_pipeline, "save_node", AsyncMock())
    monkeypatch.setattr(
        brain_pipeline, "get_genealogist_llm_client", lambda: MagicMock()
    )

    await brain_pipeline.run_brain_pipeline("x", "https://e.com", MagicMock())

    assert captured_initial["source_category"] is None
    assert captured_initial["category"] is None


# ---------------------------------------------------------------------------
# run_batch_brain_pipeline (ARG-87)
# ---------------------------------------------------------------------------


def _item(url: str = "https://example.com", content: str = "tech content") -> dict:
    return {"source_url": url, "raw_content": content, "_source": "test"}


def _batch_state(**overrides) -> dict:
    base = {
        "raw_text": "x",
        "source_url": "https://example.com",
        "is_valid": True,
        "trust_score": 0.7,
        "summary": "s",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_batch_pipeline_empty_items():
    session = MagicMock()
    result = await brain_pipeline.run_batch_brain_pipeline([], session)
    assert result == []


@pytest.mark.asyncio
async def test_batch_pipeline_cold_start_skips_genealogy(monkeypatch):
    """All items cold-start: genealogist must never run."""
    cold = _batch_state(genealogy_skipped=True, genealogy_skip_reason="cold_start")
    saved = {**cold, "saved": True}

    monkeypatch.setattr(
        brain_pipeline, "batch_triage_states", AsyncMock(return_value=[cold])
    )
    monkeypatch.setattr(
        brain_pipeline,
        "batch_embed_and_search_node",
        AsyncMock(return_value=[cold]),
    )
    genealogist_mock = AsyncMock()
    monkeypatch.setattr(brain_pipeline, "genealogist_node", genealogist_mock)
    monkeypatch.setattr(brain_pipeline, "save_node", AsyncMock(return_value=saved))
    monkeypatch.setattr(brain_pipeline, "get_genealogist_llm_client", lambda: MagicMock())

    session = MagicMock()
    session.begin_nested = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))

    results = await brain_pipeline.run_batch_brain_pipeline([_item()], session)

    assert len(results) == 1
    assert results[0]["saved"] is True
    genealogist_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_pipeline_trust_gate_skips_genealogy(monkeypatch):
    """Items below trust_skip_threshold skip genealogy with reason='low_trust'."""
    low_trust = _batch_state(trust_score=0.2, related_tech_ids=["abc"])

    monkeypatch.setattr(
        brain_pipeline, "batch_triage_states", AsyncMock(return_value=[low_trust])
    )
    monkeypatch.setattr(
        brain_pipeline,
        "batch_embed_and_search_node",
        AsyncMock(return_value=[low_trust]),
    )
    genealogist_mock = AsyncMock()
    monkeypatch.setattr(brain_pipeline, "genealogist_node", genealogist_mock)
    # save_node returns whatever state it receives, marked saved
    async def _fake_save(state, *, session):
        return {**state, "saved": True}
    monkeypatch.setattr(brain_pipeline, "save_node", _fake_save)
    monkeypatch.setattr(brain_pipeline, "get_genealogist_llm_client", lambda: MagicMock())

    session = MagicMock()
    session.begin_nested = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))

    results = await brain_pipeline.run_batch_brain_pipeline([_item()], session)

    assert results[0]["genealogy_skip_reason"] == "low_trust"
    genealogist_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_pipeline_runs_genealogy_for_high_trust(monkeypatch):
    """Items above trust threshold with related_tech_ids reach the genealogist."""
    high_trust = _batch_state(trust_score=0.8, related_tech_ids=["abc"])
    genealogized = {**high_trust, "succession_result": {"relation_type": "Enhance"}}
    saved = {**genealogized, "saved": True}

    monkeypatch.setattr(
        brain_pipeline, "batch_triage_states", AsyncMock(return_value=[high_trust])
    )
    monkeypatch.setattr(
        brain_pipeline,
        "batch_embed_and_search_node",
        AsyncMock(return_value=[high_trust]),
    )
    genealogist_mock = AsyncMock(return_value=genealogized)
    monkeypatch.setattr(brain_pipeline, "genealogist_node", genealogist_mock)
    monkeypatch.setattr(brain_pipeline, "save_node", AsyncMock(return_value=saved))

    class _FakeClient:
        async def prewarm(self, role):
            pass
        async def unload(self, role):
            pass

    monkeypatch.setattr(brain_pipeline, "get_genealogist_llm_client", lambda: _FakeClient())

    session = MagicMock()
    session.begin_nested = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))

    results = await brain_pipeline.run_batch_brain_pipeline([_item()], session)

    assert results[0]["saved"] is True
    genealogist_mock.assert_awaited_once()
