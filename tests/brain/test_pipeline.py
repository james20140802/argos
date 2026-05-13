"""Tests for argos.brain.pipeline.run_brain_pipeline (ARG-39, ARG-54).

Focus: verify that the 32B prewarm and the genealogist LLM call are skipped
when embed_and_search_node flags a cold start. We mock every node so the test
runs without Docker or Ollama.
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

    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: _FakeClient())

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
    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: _FakeClient())

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

    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: _FakeClient())

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
        brain_pipeline, "get_llm_client", lambda: MagicMock()
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
        brain_pipeline, "get_llm_client", lambda: MagicMock()
    )

    await brain_pipeline.run_brain_pipeline("x", "https://e.com", MagicMock())

    assert captured_initial["source_category"] is None
    assert captured_initial["category"] is None
