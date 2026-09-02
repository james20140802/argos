import math
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain.event_backfill import BackfillDoc, plan_backfill
from argos.brain.event_candidates import CandidateNeighbor
from argos.brain.event_scoring import DocumentFeatures
from argos.config import settings

AT = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _doc(embedding, names, at, title="t", summary="s") -> BackfillDoc:
    return BackfillDoc(
        tech_item_id=uuid.uuid4(),
        features=DocumentFeatures(
            embedding=tuple(embedding),
            names=frozenset(names),
            at=at,
            keywords=frozenset(summary.split()),
        ),
        title=title,
        summary=summary,
    )


def _session_without_db_neighbours() -> MagicMock:
    """``db_candidate_source``가 항상 빈 목록을 주는 세션 목.

    ``begin_nested``는 async context manager여야 한다 — 뒤 태스크의 실행·
    재명명 경로가 ``async with session.begin_nested():``를 쓴다.
    """

    @asynccontextmanager
    async def _nested():
        yield None

    session = MagicMock()
    session.begin_nested = MagicMock(side_effect=lambda: _nested())
    return session


@pytest.mark.asyncio
async def test_two_similar_documents_land_in_one_virtual_event(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    docs = [
        _doc([1.0, 0.0], ["anthropic"], AT, summary="claude 5 shipped"),
        _doc([1.0, 0.0], ["anthropic"], AT + timedelta(hours=2), summary="claude 5 shipped"),
    ]
    plan = await plan_backfill(
        _session_without_db_neighbours(), docs, config=settings.user.event_detection
    )
    assert plan.new_event_count == 1
    assert len({a.event_id for a in plan.assignments}) == 1
    assert [a.created for a in plan.assignments] == [True, False]


@pytest.mark.asyncio
async def test_unrelated_documents_create_separate_events(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    docs = [
        _doc([1.0, 0.0], ["anthropic"], AT, summary="claude shipped"),
        _doc([0.0, 1.0], ["mistral"], AT + timedelta(days=13), summary="totally other news"),
    ]
    plan = await plan_backfill(
        _session_without_db_neighbours(), docs, config=settings.user.event_detection
    )
    assert plan.new_event_count == 2
    assert plan.size_distribution == {1: 2}


@pytest.mark.asyncio
async def test_plan_never_writes_to_the_session(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    session = _session_without_db_neighbours()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    await plan_backfill(
        session,
        [_doc([1.0, 0.0], ["anthropic"], AT)],
        config=settings.user.event_detection,
    )
    session.add.assert_not_called()
    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_size_distribution_counts_events_by_document_count(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    docs = [
        _doc([1.0, 0.0], ["anthropic"], AT, summary="claude shipped"),
        _doc([1.0, 0.0], ["anthropic"], AT + timedelta(hours=1), summary="claude shipped"),
        _doc([0.0, 1.0], ["mistral"], AT + timedelta(days=13), summary="other news here"),
    ]
    plan = await plan_backfill(
        _session_without_db_neighbours(), docs, config=settings.user.event_detection
    )
    assert plan.size_distribution == {2: 1, 1: 1}


@pytest.mark.asyncio
async def test_pending_candidates_are_capped_at_candidate_k_by_cosine(monkeypatch):
    """오버레이 후보가 ``candidate_k``를 넘으면, DB 쿼리(``LIMIT``)와 같은
    규칙(코사인 상위 K)으로 다시 잘라야 한다 — 안 그러면 밀집한 시간 창에서
    미리보기가 실제 실행보다 더 많은 이웃을 보고, 실행이라면 갈랐을 두 문서를
    미리보기가 묶어 버린다. ``candidate_k``는 설정 기본값(25)을 그대로 쓴다
    — 테스트 편의로 낮추면 이 캡이 실제로 25에서 동작하는지 확인할 수 없다.
    """
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )

    calls: list[tuple[DocumentFeatures, list[CandidateNeighbor]]] = []

    def _recording_decide_event(features, candidates, *, config):
        calls.append((features, list(candidates)))
        # 항상 새 사건을 만든다 — 그래야 앞선 문서 하나하나가 뒤 문서의
        # pending 후보로 오버레이에 쌓인다.
        return None

    monkeypatch.setattr(event_backfill, "decide_event", _recording_decide_event)

    config = settings.user.event_detection
    k = config.candidate_k

    # k+5개의 이웃 문서를 서로 다른, 알려진 코사인 유사도로 만든다 — 각도가
    # 0에서 커질수록(인덱스가 커질수록) 주제 문서와의 코사인 유사도가
    # 단조 감소한다.
    neighbour_docs = [
        _doc(
            (math.cos(i * (math.pi / 2) / (k + 5)), math.sin(i * (math.pi / 2) / (k + 5))),
            [],
            AT,
            summary=f"doc {i}",
        )
        for i in range(k + 5)
    ]
    subject = _doc([1.0, 0.0], [], AT, summary="subject")
    docs = [*neighbour_docs, subject]

    await plan_backfill(_session_without_db_neighbours(), docs, config=config)

    subject_features, subject_candidates = calls[-1]
    assert subject_features is subject.features
    assert len(subject_candidates) == k

    # 가장 가까운 k개(각도가 가장 작은, 즉 코사인 유사도가 가장 큰 앞쪽 k개)만
    # 살아남아야 한다.
    expected_ids = {doc.tech_item_id for doc in neighbour_docs[:k]}
    actual_ids = {candidate.tech_item_id for candidate in subject_candidates}
    assert actual_ids == expected_ids


def test_cap_candidates_breaks_cosine_ties_by_id_ascending():
    """코사인 유사도가 완전히 동점이면 ``tech_item_id`` 문자열 오름차순으로
    깬다 — ``_CANDIDATE_SQL``의 ``ORDER BY ..., id``와 같은 규칙이라야 같은
    입력이 두 경로(미리보기·DB 쿼리)에서 항상 같은 결과를 낸다.
    """
    from argos.brain import event_backfill

    subject = DocumentFeatures(
        embedding=(1.0, 0.0), names=frozenset(), at=AT, keywords=frozenset()
    )

    # 세 후보 모두 주제와 동일한 임베딩이라 코사인 유사도가 완전히 동점이다
    # — 유일한 차이는 tech_item_id뿐. 일부러 정렬 안 된 순서로 넣는다.
    ids = sorted([uuid.uuid4() for _ in range(3)], key=str, reverse=True)
    candidates = [
        CandidateNeighbor(
            tech_item_id=tech_item_id,
            features=DocumentFeatures(
                embedding=(1.0, 0.0), names=frozenset(), at=AT, keywords=frozenset()
            ),
            event_ids=(uuid.uuid4(),),
        )
        for tech_item_id in ids
    ]

    capped = event_backfill._cap_candidates(subject, candidates, k=2)

    expected_order = sorted(ids, key=str)[:2]
    assert [candidate.tech_item_id for candidate in capped] == expected_order


@pytest.mark.asyncio
async def test_execute_commits_every_batch(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(event_backfill, "db_candidate_source", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        event_backfill,
        "link_document_to_event",
        AsyncMock(side_effect=lambda *a, **kw: event_backfill.LinkResult(uuid.uuid4(), True)),
    )
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    docs = [_doc([1.0, 0.0], ["a"], AT + timedelta(days=i * 40)) for i in range(5)]

    result = await event_backfill.execute_backfill(
        session, docs, config=settings.user.event_detection, batch_size=2
    )

    assert result.assigned == 5
    assert result.created_events == 5
    # 2 + 2 + 1 → 배치 경계 2회 + 마지막 잔여 1회 = 3회
    assert session.commit.await_count == 3


@pytest.mark.asyncio
async def test_execute_commits_exactly_once_per_full_batch(monkeypatch):
    """4건을 batch_size=2로 나누면 2+2 — 잔여가 없으니 마지막 커밋이 중복되지
    않아야 한다. 이 경계 조건을 놓치면 이중 커밋이나 마지막 배치 누락이
    조용히 통과할 수 있다."""
    from argos.brain import event_backfill

    monkeypatch.setattr(event_backfill, "db_candidate_source", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        event_backfill,
        "link_document_to_event",
        AsyncMock(side_effect=lambda *a, **kw: event_backfill.LinkResult(uuid.uuid4(), True)),
    )
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    docs = [_doc([1.0, 0.0], ["a"], AT + timedelta(days=i * 40)) for i in range(4)]

    result = await event_backfill.execute_backfill(
        session, docs, config=settings.user.event_detection, batch_size=2
    )

    assert result.assigned == 4
    # 2 + 2, no trailing partial batch → exactly 2 commits.
    assert session.commit.await_count == 2


@pytest.mark.asyncio
async def test_execute_skips_a_failing_document_without_aborting(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(event_backfill, "db_candidate_source", AsyncMock(return_value=[]))
    calls = {"n": 0}

    async def _link(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return event_backfill.LinkResult(uuid.uuid4(), True)

    monkeypatch.setattr(event_backfill, "link_document_to_event", _link)
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    docs = [_doc([1.0, 0.0], ["a"], AT + timedelta(days=i * 40)) for i in range(3)]

    result = await event_backfill.execute_backfill(
        session, docs, config=settings.user.event_detection, batch_size=10
    )

    assert result.assigned == 2
    assert result.skipped == 1


@pytest.mark.asyncio
async def test_execute_flushes_so_the_next_document_sees_the_link(monkeypatch):
    """실행 모드는 오버레이를 쓰지 않는다 — flush한 링크를 DB 조회가 본다."""
    from argos.brain import event_backfill

    monkeypatch.setattr(event_backfill, "db_candidate_source", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        event_backfill,
        "link_document_to_event",
        AsyncMock(side_effect=lambda *a, **kw: event_backfill.LinkResult(uuid.uuid4(), True)),
    )
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    docs = [_doc([1.0, 0.0], ["a"], AT)]

    await event_backfill.execute_backfill(
        session, docs, config=settings.user.event_detection, batch_size=10
    )
    session.flush.assert_awaited()


@pytest.mark.asyncio
async def test_rename_updates_each_event_and_commits_per_batch(monkeypatch):
    from argos.brain import event_backfill
    from argos.brain.event_naming import EvidenceDoc

    applied = AsyncMock(return_value=True)
    monkeypatch.setattr(event_backfill, "apply_event_naming", applied)
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    events = [
        event_backfill.StaleEvent(
            event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")]
        )
        for _ in range(3)
    ]

    result = await event_backfill.rename_stale_events(session, events, batch_size=2)

    assert result.renamed == 3
    assert result.skipped == 0
    assert applied.await_count == 3
    assert session.commit.await_count == 2


@pytest.mark.asyncio
async def test_rename_counts_a_failed_generation_as_skipped(monkeypatch):
    from argos.brain import event_backfill
    from argos.brain.event_naming import EvidenceDoc

    monkeypatch.setattr(event_backfill, "apply_event_naming", AsyncMock(return_value=False))
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    events = [
        event_backfill.StaleEvent(
            event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")]
        )
    ]

    result = await event_backfill.rename_stale_events(session, events, batch_size=10)

    assert result.renamed == 0
    assert result.skipped == 1


@pytest.mark.asyncio
async def test_rename_forwards_the_snapshot_version_as_the_write_guard(monkeypatch):
    """스냅샷 시점의 행 버전이 apply_event_naming까지 그대로 간다 (ARG-274)."""
    from argos.brain import event_backfill
    from argos.brain.event_naming import EvidenceDoc

    applied = AsyncMock(return_value=True)
    monkeypatch.setattr(event_backfill, "apply_event_naming", applied)
    session = _session_without_db_neighbours()
    session.commit = AsyncMock()
    snapshot = datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc)

    await event_backfill.rename_stale_events(
        session,
        [
            event_backfill.StaleEvent(
                event_id=uuid.uuid4(),
                docs=[EvidenceDoc(title="t", summary="s")],
                updated_at=snapshot,
            )
        ],
        batch_size=10,
    )

    assert applied.await_args.kwargs["expected_updated_at"] == snapshot


def test_stale_event_query_reads_the_row_version_for_the_guard():
    """근거와 같은 SELECT에서 updated_at을 들고 나와야 가드를 걸 수 있다."""
    from argos.brain.event_backfill import _STALE_EVENTS_SQL

    assert "updated_at" in str(_STALE_EVENTS_SQL).lower()


def test_stale_event_defaults_to_no_guard():
    """스냅샷 없이 만든 StaleEvent는 가드를 걸지 않는다 (온라인 경로와 같은 계약)."""
    from argos.brain.event_backfill import StaleEvent

    assert StaleEvent(event_id=uuid.uuid4(), docs=[]).updated_at is None


def test_stale_event_query_excludes_tombstones_and_targets_stale_or_untitled():
    """SQL 문자열 수준 회귀 — 조건 세 개가 모두 남아 있어야 한다."""
    from argos.brain.event_backfill import _STALE_EVENTS_SQL

    sql = str(_STALE_EVENTS_SQL).lower()
    assert "naming_stale" in sql
    assert "title is null" in sql
    assert "merged_into_id is null" in sql
