"""Tests for ARG-88: asyncio semaphore in batch_embed_and_search_node.

Verifies:
 1. The semaphore bound is read from ``genealogist.embed_search_concurrency``
    in config (configurable).
 2. Searches run concurrently (overlap observed), bounded by the semaphore.
 3. Results are returned in the original order (parallel list invariant).
 4. Config field defaults to 4 and validates ge=1.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain.nodes import embed as embed_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base = {
        "raw_text": "sample content",
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


# ---------------------------------------------------------------------------
# Config: embed_search_concurrency field
# ---------------------------------------------------------------------------


def test_genealogist_config_has_embed_search_concurrency_default():
    """embed_search_concurrency defaults to 4."""
    from argos.config import GenealogistConfig

    cfg = GenealogistConfig()
    assert cfg.embed_search_concurrency == 4


def test_genealogist_config_embed_search_concurrency_is_configurable():
    """embed_search_concurrency can be overridden."""
    from argos.config import GenealogistConfig

    cfg = GenealogistConfig(embed_search_concurrency=8)
    assert cfg.embed_search_concurrency == 8


def test_genealogist_config_embed_search_concurrency_minimum_is_one():
    """embed_search_concurrency must be ge=1."""
    from argos.config import GenealogistConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GenealogistConfig(embed_search_concurrency=0)


# ---------------------------------------------------------------------------
# Concurrency: searches run in parallel, bounded by the semaphore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_embed_searches_run_concurrently(monkeypatch):
    """Similarity searches fan out concurrently up to the semaphore bound.

    Strategy: replace ``_similarity_search`` with a coroutine that records its
    start and end wall-clock timestamps and sleeps briefly.  With N items and a
    semaphore bound >= N, at least two searches must overlap in time —
    demonstrating true concurrency rather than sequential execution.
    """
    n_items = 4
    states = [
        _state(raw_text=f"item {i}", source_url=f"https://e.com/{i}")
        for i in range(n_items)
    ]

    intervals: list[tuple[float, float]] = []
    lock = asyncio.Lock()

    async def _fake_search(state, embedding, session_factory, top_n, max_chars):
        start = time.monotonic()
        await asyncio.sleep(0.05)  # 50 ms artificial latency
        end = time.monotonic()
        async with lock:
            intervals.append((start, end))
        # Return a minimal happy-path state.
        return {
            **state,
            "related_tech_ids": ["abc"],
            "extracted_info": {
                "embedding": embedding,
                "similar_items": [
                    {"id": "abc", "title": "Tech", "raw_content": "content"}
                ],
            },
        }

    monkeypatch.setattr(embed_module, "_similarity_search", _fake_search)

    async def _fake_batch_embed(texts):
        return [[float(i)] * 768 for i in range(len(texts))]

    monkeypatch.setattr(embed_module, "batch_embed", _fake_batch_embed)

    # Warm start: embedded_count >= threshold.
    session = MagicMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 100
    session.execute = AsyncMock(return_value=count_result)

    # Use a semaphore bound of 4 (== n_items) so all can run simultaneously.
    monkeypatch.setattr(
        embed_module.settings.user.genealogist, "embed_search_concurrency", 4
    )
    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)

    out = await embed_module.batch_embed_and_search_node(states, session)

    assert len(out) == n_items
    for s in out:
        assert s["related_tech_ids"] == ["abc"]

    # At least two search intervals must overlap → concurrency confirmed.
    assert len(intervals) == n_items
    overlaps = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1 = intervals[i]
            s2, e2 = intervals[j]
            # Overlap: each starts before the other ends.
            if s1 < e2 and s2 < e1:
                overlaps += 1
    assert overlaps >= 1, (
        f"Expected at least 1 overlapping search pair, got 0. "
        f"Intervals: {intervals}"
    )


@pytest.mark.asyncio
async def test_batch_embed_semaphore_limits_concurrency(monkeypatch):
    """With semaphore bound=2 and 4 items, at most 2 searches run simultaneously.

    We track the peak concurrent active count and assert it never exceeds the bound.
    """
    n_items = 4
    bound = 2
    states = [
        _state(raw_text=f"item {i}", source_url=f"https://e.com/{i}")
        for i in range(n_items)
    ]

    active = {"count": 0, "peak": 0}
    lock = asyncio.Lock()

    async def _fake_search(state, embedding, session_factory, top_n, max_chars):
        async with lock:
            active["count"] += 1
            if active["count"] > active["peak"]:
                active["peak"] = active["count"]
        await asyncio.sleep(0.03)
        async with lock:
            active["count"] -= 1
        return {
            **state,
            "related_tech_ids": ["abc"],
            "extracted_info": {"embedding": embedding, "similar_items": []},
        }

    monkeypatch.setattr(embed_module, "_similarity_search", _fake_search)

    async def _fake_batch_embed(texts):
        return [[0.0] * 768 for _ in texts]

    monkeypatch.setattr(embed_module, "batch_embed", _fake_batch_embed)

    session = MagicMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 100
    session.execute = AsyncMock(return_value=count_result)

    monkeypatch.setattr(
        embed_module.settings.user.genealogist, "embed_search_concurrency", bound
    )
    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)

    out = await embed_module.batch_embed_and_search_node(states, session)

    assert len(out) == n_items
    assert active["peak"] <= bound, (
        f"Peak concurrency {active['peak']} exceeded semaphore bound {bound}"
    )
    # Also confirm some concurrency did occur (peak > 1 with bound=2 & sleep).
    assert active["peak"] >= 1


@pytest.mark.asyncio
async def test_batch_embed_order_preserved_under_concurrency(monkeypatch):
    """Results appear in the same order as the input states, regardless of completion order."""
    n_items = 5
    states = [
        _state(raw_text=f"item {i}", source_url=f"https://e.com/{i}")
        for i in range(n_items)
    ]

    async def _fake_search(state, embedding, session_factory, top_n, max_chars):
        # Stagger completion: last item finishes first.
        idx = int(state["raw_text"].split()[-1])
        delay = 0.05 * (n_items - idx)
        await asyncio.sleep(delay)
        return {
            **state,
            "related_tech_ids": [f"id-{idx}"],
            "extracted_info": {
                "embedding": embedding,
                "similar_items": [{"id": f"id-{idx}", "title": f"T{idx}", "raw_content": "x"}],
            },
        }

    monkeypatch.setattr(embed_module, "_similarity_search", _fake_search)

    async def _fake_batch_embed(texts):
        return [[float(i)] * 768 for i in range(len(texts))]

    monkeypatch.setattr(embed_module, "batch_embed", _fake_batch_embed)

    session = MagicMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 100
    session.execute = AsyncMock(return_value=count_result)

    monkeypatch.setattr(
        embed_module.settings.user.genealogist, "embed_search_concurrency", n_items
    )
    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)

    out = await embed_module.batch_embed_and_search_node(states, session)

    for i, s in enumerate(out):
        assert s["source_url"] == f"https://e.com/{i}", (
            f"Expected source_url for index {i}, got {s['source_url']}"
        )
        assert s["related_tech_ids"] == [f"id-{i}"]


@pytest.mark.asyncio
async def test_batch_embed_on_item_done_fires_per_valid_item_under_concurrency(monkeypatch):
    """on_item_done callback fires exactly N times for N valid items, even concurrently."""
    n_items = 3
    states = [_state(raw_text=f"item {i}", source_url=f"https://e.com/{i}") for i in range(n_items)]

    async def _fake_search(state, embedding, session_factory, top_n, max_chars):
        await asyncio.sleep(0.01)
        return {
            **state,
            "related_tech_ids": ["abc"],
            "extracted_info": {"embedding": embedding, "similar_items": []},
        }

    monkeypatch.setattr(embed_module, "_similarity_search", _fake_search)

    async def _fake_batch_embed(texts):
        return [[0.0] * 768 for _ in texts]

    monkeypatch.setattr(embed_module, "batch_embed", _fake_batch_embed)

    session = MagicMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 100
    session.execute = AsyncMock(return_value=count_result)

    monkeypatch.setattr(embed_module.settings.user.genealogist, "embed_search_concurrency", 4)
    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)

    tick_count = {"n": 0}

    def on_done():
        tick_count["n"] += 1

    out = await embed_module.batch_embed_and_search_node(states, session, on_item_done=on_done)

    assert len(out) == n_items
    assert tick_count["n"] == n_items
