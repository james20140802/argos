import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain.event_assignment import (
    LinkResult,
    db_candidate_source,
    decide_event,
    link_document_to_event,
)
from argos.brain.event_candidates import CandidateNeighbor
from argos.brain.event_scoring import DocumentFeatures
from argos.config import EventDetectionConfig, settings

AT = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _features(embedding, names=(), at=AT, keywords=()) -> DocumentFeatures:
    return DocumentFeatures(
        embedding=tuple(embedding),
        names=frozenset(names),
        at=at,
        keywords=frozenset(keywords),
    )


def test_decide_event_returns_the_event_above_threshold():
    config = settings.user.event_detection
    event_id = uuid.uuid4()
    subject = _features([1.0, 0.0], names=["anthropic"], keywords=["claude"])
    neighbour = CandidateNeighbor(
        tech_item_id=uuid.uuid4(),
        features=_features([1.0, 0.0], names=["anthropic"], keywords=["claude"]),
        event_ids=(event_id,),
    )
    assert decide_event(subject, [neighbour], config=config) == event_id


def test_decide_event_returns_none_below_threshold():
    config = settings.user.event_detection
    subject = _features([1.0, 0.0], names=["anthropic"], keywords=["claude"])
    neighbour = CandidateNeighbor(
        tech_item_id=uuid.uuid4(),
        features=_features(
            [0.0, 1.0], names=["unrelated"], at=AT - timedelta(days=13), keywords=["other"]
        ),
        event_ids=(uuid.uuid4(),),
    )
    assert decide_event(subject, [neighbour], config=config) is None


def test_decide_event_ignores_neighbours_without_an_event():
    config = settings.user.event_detection
    subject = _features([1.0, 0.0])
    neighbour = CandidateNeighbor(
        tech_item_id=uuid.uuid4(), features=_features([1.0, 0.0]), event_ids=()
    )
    assert decide_event(subject, [neighbour], config=config) is None


@pytest.mark.asyncio
async def test_db_candidate_source_forwards_the_callers_config(monkeypatch):
    """``db_candidate_source`` must not fall back to ``fetch_candidates``'s
    global-settings default for window_days/limit — it has to forward
    whatever config the caller is scoring with. Otherwise a caller with its
    own ``EventDetectionConfig`` (e.g. a future window override) would fetch
    candidates from a different window than ``decide_event``'s time decay
    and ``_cap_candidates``'s cap believe they are working under, and the
    preview's printed thresholds would stop describing the candidates
    actually fetched."""
    default = settings.user.event_detection
    custom_config = EventDetectionConfig(
        window_days=default.window_days + 1,
        candidate_k=default.candidate_k + 1,
    )
    assert custom_config.window_days != default.window_days
    assert custom_config.candidate_k != default.candidate_k

    recorded_kwargs = {}

    async def _fake_fetch_candidates(session, **kwargs):
        recorded_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(
        "argos.brain.event_assignment.fetch_candidates", _fake_fetch_candidates
    )

    session = MagicMock()

    @asynccontextmanager
    async def _begin_nested():
        yield None

    session.begin_nested = MagicMock(side_effect=lambda: _begin_nested())

    result = await db_candidate_source(
        session, embedding=[1.0, 0.0], at=AT, config=custom_config
    )

    assert result == []
    assert recorded_kwargs["window_days"] == custom_config.window_days
    assert recorded_kwargs["limit"] == custom_config.candidate_k


def _session() -> MagicMock:
    session = MagicMock()
    session.add = MagicMock()
    session.execute = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_link_creates_a_new_event_marked_naming_stale():
    session = _session()
    item_id = uuid.uuid4()
    result = await link_document_to_event(
        session, tech_item_id=item_id, event_id=None, occurred_at=AT
    )
    assert isinstance(result, LinkResult)
    assert result.created is True
    created_event = session.add.call_args.args[0]
    assert created_event.id == result.event_id
    assert created_event.occurred_at == AT
    assert created_event.naming_stale is True
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_link_to_an_existing_event_sets_naming_stale():
    session = _session()
    event_id = uuid.uuid4()
    result = await link_document_to_event(
        session, tech_item_id=uuid.uuid4(), event_id=event_id, occurred_at=AT
    )
    assert result == LinkResult(event_id=event_id, created=False)
    session.add.assert_not_called()
    # 링크 INSERT 한 번 + naming_stale UPDATE 한 번.
    assert session.execute.await_count == 2


@pytest.mark.asyncio
async def test_link_is_idempotent_on_conflict():
    """같은 (사건, 문서) 쌍을 두 번 써도 예외 없이 통과한다 — 재실행 멱등성."""
    session = _session()
    event_id = uuid.uuid4()
    item_id = uuid.uuid4()
    first = await link_document_to_event(
        session, tech_item_id=item_id, event_id=event_id, occurred_at=AT
    )
    second = await link_document_to_event(
        session, tech_item_id=item_id, event_id=event_id, occurred_at=AT
    )
    assert first == second == LinkResult(event_id=event_id, created=False)
    insert_statements = [call.args[0] for call in session.execute.await_args_list]
    assert any("ON CONFLICT" in str(stmt).upper() for stmt in insert_statements)
