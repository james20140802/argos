"""배정 노드의 제어 흐름 — ARG-266. fetch_candidates를 갈아 끼워 DB 없이 돈다."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from argos.brain.event_candidates import CandidateNeighbor
from argos.brain.event_scoring import DocumentFeatures
from argos.brain.nodes import assign_event as assign_event_module
from argos.brain.nodes.assign_event import assign_event_node


def _state(**overrides) -> dict:
    base = {
        "raw_text": "x",
        "source_url": "https://example.com/a",
        "is_valid": True,
        "trust_score": 0.7,
        "summary": "a short summary",
        "extracted_info": {"embedding": [1.0, 0.0, 0.0]},
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
        "published_at": datetime(2026, 8, 20, tzinfo=timezone.utc),
        "entity_names": ["acme corp"],
    }
    base.update(overrides)
    return base


def _features(**overrides) -> DocumentFeatures:
    base = dict(
        embedding=(1.0, 0.0, 0.0),
        names=frozenset({"acme corp"}),
        at=datetime(2026, 8, 20, tzinfo=timezone.utc),
        keywords=frozenset(),
    )
    base.update(overrides)
    return DocumentFeatures(**base)


@pytest.mark.asyncio
async def test_a_close_neighbour_hands_over_its_event(monkeypatch):
    event_id = uuid.uuid4()
    candidate = CandidateNeighbor(
        tech_item_id=uuid.uuid4(),
        features=_features(),
        event_ids=(event_id,),
    )
    monkeypatch.setattr(
        assign_event_module, "fetch_candidates", AsyncMock(return_value=[candidate])
    )

    result = await assign_event_node(_state(), session=AsyncMock())

    assert result["event_id"] == event_id


@pytest.mark.asyncio
async def test_a_distant_neighbour_leaves_the_event_open(monkeypatch):
    event_id = uuid.uuid4()
    candidate = CandidateNeighbor(
        tech_item_id=uuid.uuid4(),
        features=_features(
            embedding=(0.0, 1.0, 0.0),
            names=frozenset({"unrelated corp"}),
            keywords=frozenset({"totally", "different"}),
        ),
        event_ids=(event_id,),
    )
    monkeypatch.setattr(
        assign_event_module, "fetch_candidates", AsyncMock(return_value=[candidate])
    )

    result = await assign_event_node(_state(), session=AsyncMock())

    assert result["event_id"] is None


@pytest.mark.asyncio
async def test_no_neighbours_leaves_the_event_open(monkeypatch):
    monkeypatch.setattr(
        assign_event_module, "fetch_candidates", AsyncMock(return_value=[])
    )

    result = await assign_event_node(_state(), session=AsyncMock())

    assert result["event_id"] is None


@pytest.mark.asyncio
async def test_a_failing_candidate_query_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        assign_event_module,
        "fetch_candidates",
        AsyncMock(side_effect=RuntimeError("db exploded")),
    )

    result = await assign_event_node(_state(), session=AsyncMock())

    assert result["event_id"] is None


@pytest.mark.asyncio
async def test_an_invalid_state_is_passed_through(monkeypatch):
    fetch_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(assign_event_module, "fetch_candidates", fetch_mock)

    result = await assign_event_node(_state(is_valid=False), session=AsyncMock())

    fetch_mock.assert_not_called()
    assert "event_id" not in result


@pytest.mark.asyncio
async def test_a_document_without_an_embedding_is_passed_through(monkeypatch):
    fetch_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(assign_event_module, "fetch_candidates", fetch_mock)

    result = await assign_event_node(
        _state(extracted_info={}), session=AsyncMock()
    )

    fetch_mock.assert_not_called()
    assert result["event_id"] is None
