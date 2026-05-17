"""Tests for ARG-99: on_item_done callback parameters on pipeline brain nodes.

Verifies that `batch_triage_states`, `batch_embed_and_search_node`, and the
genealogy loop inside `run_batch_brain_pipeline` accept an optional
`on_item_done` callable and invoke it once per processed item.

The callback interface is the prerequisite for the Rich progress bar wired in
the CLI (ARG-101). All callbacks default to ``None``, so existing callers and
existing tests must continue to work unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain import pipeline as brain_pipeline
from argos.brain.nodes import triage as triage_module
from argos.brain.nodes import embed as embed_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
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


def _item(url: str) -> dict:
    return {"source_url": url, "raw_content": "content", "_source": "test"}


def _nested_cm():
    """Build a session.begin_nested context manager mock."""
    return MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))


# ---------------------------------------------------------------------------
# batch_triage_states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_triage_states_invokes_callback_once_per_item(monkeypatch):
    """Callback fires exactly N times for N input states."""
    states = [_state(source_url=f"https://e.com/{i}") for i in range(3)]

    async def _fake_triage_one(state, client, keep_alive):
        return state

    monkeypatch.setattr(triage_module, "_triage_one", _fake_triage_one)

    class _FakeClient:
        async def unload(self, role):  # noqa: ARG002
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _FakeClient())

    calls = {"n": 0}

    def on_done():
        calls["n"] += 1

    results = await triage_module.batch_triage_states(states, on_item_done=on_done)

    assert len(results) == 3
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_batch_triage_states_default_callback_is_none(monkeypatch):
    """Without a callback, behavior is unchanged (back-compat)."""
    states = [_state()]

    async def _fake_triage_one(state, client, keep_alive):
        return state

    monkeypatch.setattr(triage_module, "_triage_one", _fake_triage_one)

    class _FakeClient:
        async def unload(self, role):  # noqa: ARG002
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _FakeClient())

    # No on_item_done passed — must not raise.
    results = await triage_module.batch_triage_states(states)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_batch_triage_states_empty_does_not_invoke_callback():
    """Empty input → callback never fires."""
    calls = {"n": 0}

    def on_done():
        calls["n"] += 1

    results = await triage_module.batch_triage_states([], on_item_done=on_done)
    assert results == []
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# batch_embed_and_search_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_embed_invokes_callback_per_valid_item(monkeypatch):
    """Callback fires once per *valid* item processed by the embed loop.

    Invalid items are pass-through and represent no work, so they don't tick.
    """
    valid_states = [_state(is_valid=True, raw_text=f"text-{i}") for i in range(2)]
    invalid_state = _state(is_valid=False)
    states = [valid_states[0], invalid_state, valid_states[1]]

    async def _fake_batch_embed(texts):
        return [[0.0] * 768 for _ in texts]

    monkeypatch.setattr(embed_module, "batch_embed", _fake_batch_embed)

    # Force cold-start to skip the similarity query entirely.
    session = MagicMock()

    async def _exec(_stmt, _params=None):
        result = MagicMock()
        result.scalar.return_value = 0  # 0 embedded items → cold_start
        return result

    session.execute = _exec

    calls = {"n": 0}

    def on_done():
        calls["n"] += 1

    out = await embed_module.batch_embed_and_search_node(
        states, session, on_item_done=on_done
    )

    assert len(out) == 3
    assert calls["n"] == 2  # only the two valid items ticked


@pytest.mark.asyncio
async def test_batch_embed_default_callback_is_none(monkeypatch):
    """No callback → no behavior change."""
    states = [_state(is_valid=True)]

    async def _fake_batch_embed(texts):
        return [[0.0] * 768 for _ in texts]

    monkeypatch.setattr(embed_module, "batch_embed", _fake_batch_embed)
    session = MagicMock()

    async def _exec(_stmt, _params=None):
        result = MagicMock()
        result.scalar.return_value = 0
        return result

    session.execute = _exec

    out = await embed_module.batch_embed_and_search_node(states, session)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_batch_embed_no_valid_items_does_not_invoke_callback():
    """All invalid → callback never fires (and embed isn't called)."""
    calls = {"n": 0}
    session = MagicMock()
    out = await embed_module.batch_embed_and_search_node(
        [_state(is_valid=False)],
        session,
        on_item_done=lambda: calls.__setitem__("n", calls["n"] + 1),
    )
    assert len(out) == 1
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# Genealogy loop in run_batch_brain_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genealogy_loop_invokes_callback_per_candidate(monkeypatch):
    """The genealogy loop fires on_genealogy_item_done once per candidate."""
    high_a = _state(trust_score=0.8, related_tech_ids=["a"], source_url="https://e.com/a")
    high_b = _state(trust_score=0.8, related_tech_ids=["b"], source_url="https://e.com/b")

    monkeypatch.setattr(
        brain_pipeline,
        "batch_triage_states",
        AsyncMock(return_value=[high_a, high_b]),
    )
    monkeypatch.setattr(
        brain_pipeline,
        "batch_embed_and_search_node",
        AsyncMock(return_value=[high_a, high_b]),
    )

    async def _fake_geno(state, *, prewarm_task=None):  # noqa: ARG001
        return {**state, "succession_result": {"relation_type": "Enhance"}}

    monkeypatch.setattr(brain_pipeline, "genealogist_node", _fake_geno)

    async def _fake_save(state, *, session):  # noqa: ARG001
        return {**state, "saved": True}

    monkeypatch.setattr(brain_pipeline, "save_node", _fake_save)

    class _FakeClient:
        async def prewarm(self, role):  # noqa: ARG002
            return None

        async def unload(self, role):  # noqa: ARG002
            return None

    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: _FakeClient())

    session = MagicMock()
    session.begin_nested = _nested_cm()

    calls = {"n": 0}

    results = await brain_pipeline.run_batch_brain_pipeline(
        [_item("https://e.com/a"), _item("https://e.com/b")],
        session,
        on_genealogy_item_done=lambda: calls.__setitem__("n", calls["n"] + 1),
    )

    assert len(results) == 2
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_run_batch_brain_pipeline_forwards_triage_and_embed_callbacks(monkeypatch):
    """The pipeline forwards on_triage_item_done and on_embed_item_done."""
    captured = {"triage": None, "embed": None}

    async def _fake_triage(states, *, on_item_done=None):
        captured["triage"] = on_item_done
        return [_state(is_valid=False)]

    async def _fake_embed(states, session, *, on_item_done=None):  # noqa: ARG001
        captured["embed"] = on_item_done
        return list(states)

    async def _fake_save(state, *, session):  # noqa: ARG001
        return {**state, "saved": False}

    monkeypatch.setattr(brain_pipeline, "batch_triage_states", _fake_triage)
    monkeypatch.setattr(brain_pipeline, "batch_embed_and_search_node", _fake_embed)
    monkeypatch.setattr(brain_pipeline, "save_node", _fake_save)
    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: MagicMock())

    session = MagicMock()
    session.begin_nested = _nested_cm()

    triage_cb = lambda: None  # noqa: E731
    embed_cb = lambda: None  # noqa: E731

    await brain_pipeline.run_batch_brain_pipeline(
        [_item("https://e.com/x")],
        session,
        on_triage_item_done=triage_cb,
        on_embed_item_done=embed_cb,
    )

    assert captured["triage"] is triage_cb
    assert captured["embed"] is embed_cb


@pytest.mark.asyncio
async def test_save_loop_invokes_callback_per_state(monkeypatch):
    """on_save_item_done fires once per state in the save loop (ARG-101 wiring).

    The save loop already iterates every brain state — invalid ones skip the
    actual save_node call but still represent a slot in the bar.
    """
    # Cold-start both items so the genealogy branch (and 32B prewarm) is
    # short-circuited — keeps this focused on the save loop.
    high = _state(
        trust_score=0.8,
        related_tech_ids=[],
        genealogy_skipped=True,
        genealogy_skip_reason="cold_start",
    )
    invalid = _state(is_valid=False, source_url="https://e.com/bad")

    monkeypatch.setattr(
        brain_pipeline,
        "batch_triage_states",
        AsyncMock(return_value=[high, invalid]),
    )
    monkeypatch.setattr(
        brain_pipeline,
        "batch_embed_and_search_node",
        AsyncMock(return_value=[high, invalid]),
    )
    monkeypatch.setattr(brain_pipeline, "genealogist_node", AsyncMock(return_value=high))

    async def _fake_save(state, *, session):  # noqa: ARG001
        return {**state, "saved": True}

    monkeypatch.setattr(brain_pipeline, "save_node", _fake_save)
    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: MagicMock())

    session = MagicMock()
    session.begin_nested = _nested_cm()

    calls = {"n": 0}

    await brain_pipeline.run_batch_brain_pipeline(
        [_item("https://e.com/a"), _item("https://e.com/b")],
        session,
        on_save_item_done=lambda: calls.__setitem__("n", calls["n"] + 1),
    )

    # Save loop iterates 2 states (high, invalid).
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_run_batch_brain_pipeline_no_callbacks_default_none(monkeypatch):
    """Calling without callback kwargs keeps the original behavior unchanged."""
    captured = {"triage": "<unset>", "embed": "<unset>"}

    async def _fake_triage(states, *, on_item_done=None):
        captured["triage"] = on_item_done
        return [_state(is_valid=False)]

    async def _fake_embed(states, session, *, on_item_done=None):  # noqa: ARG001
        captured["embed"] = on_item_done
        return list(states)

    async def _fake_save(state, *, session):  # noqa: ARG001
        return {**state, "saved": False}

    monkeypatch.setattr(brain_pipeline, "batch_triage_states", _fake_triage)
    monkeypatch.setattr(brain_pipeline, "batch_embed_and_search_node", _fake_embed)
    monkeypatch.setattr(brain_pipeline, "save_node", _fake_save)
    monkeypatch.setattr(brain_pipeline, "get_llm_client", lambda: MagicMock())

    session = MagicMock()
    session.begin_nested = _nested_cm()

    await brain_pipeline.run_batch_brain_pipeline(
        [_item("https://e.com/x")], session
    )

    assert captured["triage"] is None
    assert captured["embed"] is None
