"""배정 노드의 제어 흐름 — ARG-266. fetch_candidates를 갈아 끼워 DB 없이 돈다."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain import event_assignment as event_assignment_module
from argos.brain.event_candidates import CandidateNeighbor
from argos.brain.event_scoring import DocumentFeatures
from argos.brain.nodes.assign_event import assign_event_node


def _session() -> MagicMock:
    """assign_event_node가 후보 조회를 ``session.begin_nested()`` 세이브포인트
    안에서 부르므로(ARG-266 C2), 그 async context manager 프로토콜을 실제로
    지원하는 mock이 필요하다 — 맨 ``AsyncMock()``은 ``begin_nested()`` 호출
    자체를 코루틴으로 취급해 ``async with``에서 깨진다."""
    session = MagicMock()
    session.begin_nested = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return session


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
        event_assignment_module, "fetch_candidates", AsyncMock(return_value=[candidate])
    )

    result = await assign_event_node(_state(), session=_session())

    assert result["event_id"] == event_id
    # 판정이 끝까지 돌았다 — save_node가 이 event_id를 그대로 신뢰해도 된다.
    assert result["event_assigned"] is True


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
        event_assignment_module, "fetch_candidates", AsyncMock(return_value=[candidate])
    )

    result = await assign_event_node(_state(), session=_session())

    assert result["event_id"] is None
    # 판정은 끝까지 돌았다 — event_id=None은 "새 사건이 필요하다"는 뜻이지
    # 실패가 아니다. save_node가 새 사건을 만들어도 되는 경우.
    assert result["event_assigned"] is True


@pytest.mark.asyncio
async def test_no_neighbours_leaves_the_event_open(monkeypatch):
    monkeypatch.setattr(
        event_assignment_module, "fetch_candidates", AsyncMock(return_value=[])
    )

    result = await assign_event_node(_state(), session=_session())

    assert result["event_id"] is None
    assert result["event_assigned"] is True


@pytest.mark.asyncio
async def test_a_failing_candidate_query_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        event_assignment_module,
        "fetch_candidates",
        AsyncMock(side_effect=RuntimeError("db exploded")),
    )

    result = await assign_event_node(_state(), session=_session())

    assert result["event_id"] is None
    # ARG-266 C1: 실패했을 때는 event_assigned=False여야 한다 — save_node가
    # 이걸 "못 찾음"과 헷갈려 새 사건을 만들면, 배정이 조직적으로 실패하는
    # 사고가 문서마다 잘못된 사건을 하나씩 영구히 남긴다.
    assert result["event_assigned"] is False


@pytest.mark.asyncio
async def test_an_invalid_state_is_passed_through(monkeypatch):
    fetch_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(event_assignment_module, "fetch_candidates", fetch_mock)

    result = await assign_event_node(_state(is_valid=False), session=_session())

    fetch_mock.assert_not_called()
    assert "event_id" not in result
    assert "event_assigned" not in result


@pytest.mark.asyncio
async def test_a_document_without_an_embedding_is_passed_through(monkeypatch):
    fetch_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(event_assignment_module, "fetch_candidates", fetch_mock)

    result = await assign_event_node(
        _state(extracted_info={}), session=_session()
    )

    fetch_mock.assert_not_called()
    assert result["event_id"] is None
    # 시도조차 안 했다 — "실패"와 같은 신호(False)를 써서 save_node가 새
    # 사건을 만들지 않게 한다. (임베딩 없는 문서는 애초에 후보 비교가
    # 불가능하므로, 이 경로는 "판정 완료, 못 찾음"이 아니다.)
    assert result["event_assigned"] is False
