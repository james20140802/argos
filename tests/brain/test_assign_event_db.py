"""save_node의 사건 쓰기 — ARG-266 DB 통합 테스트. 부모 AC를 종단으로 확인한다.

패턴은 ``tests/brain/test_event_candidates_db.py``와 같다: 모듈 스코프
``session_factory`` 픽스처(NullPool) + 이 모듈이 만든 행만 정리 + Postgres가
없으면 통째로 skip.

노드를 따로 부르지 않고 ``argos.brain.pipeline._assign_then_save``를 그대로
쓴다 — 실제 파이프라인이 부르는 자리와 같은 함수여야 "배정 뒤 저장"이라는
계약을 종단으로 검증한 것이 된다.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain import pipeline as brain_pipeline
from argos.brain.pipeline import _assign_then_save
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-266-assign-event-test.example.com/"

NOW = datetime.now(timezone.utc)


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-266 assign_event DB "
            "integration test (start the Docker DB to run it)"
        )


@pytest.fixture
async def session_factory():
    """NullPool 기반 sessionmaker를 주고, 끝나면 이 파일이 만든 행만 지운다.

    event_documents를 지우기 전에 관련 event_id를 먼저 모아 둔다 — save_node가
    만드는 TechEvent에는 구분 가능한 title이 없어(occurred_at만 채운다),
    링크가 사라진 뒤에는 이 테스트가 만든 사건을 더 이상 찾을 수 없다.
    """
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with factory() as session:
            item_ids_result = await session.execute(
                TechItem.__table__.select().with_only_columns(TechItem.id).where(
                    TechItem.source_url.like(f"{_URL_PREFIX}%")
                )
            )
            item_ids = [row[0] for row in item_ids_result.all()]
            event_ids: set[uuid.UUID] = set()
            if item_ids:
                links_result = await session.execute(
                    select(EventDocument.event_id).where(
                        EventDocument.tech_item_id.in_(item_ids)
                    )
                )
                event_ids = {row[0] for row in links_result.all()}
                await session.execute(
                    delete(EventDocument).where(EventDocument.tech_item_id.in_(item_ids))
                )
                await session.execute(
                    delete(DocumentEntity).where(DocumentEntity.tech_item_id.in_(item_ids))
                )
            await session.execute(
                delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
            )
            if event_ids:
                await session.execute(delete(TechEvent).where(TechEvent.id.in_(event_ids)))
            await session.execute(
                delete(Entity).where(Entity.normalized_key.like("arg-266-test-%"))
            )
            await session.commit()
        await engine.dispose()


def _embedding(x: float, y: float) -> list[float]:
    """768차원 임베딩. 앞 두 성분만 채운다 — 코사인은 정규화되므로 크기와
    무관하게 (x, y)의 각도만으로 값이 정해진다."""
    vec = [0.0] * 768
    vec[0] = x
    vec[1] = y
    return vec


def _state(
    suffix: str,
    *,
    embedding: list[float],
    published_at: datetime,
    names: tuple[str, ...] = (),
    summary: str = "",
) -> dict:
    return {
        "raw_text": f"ARG-266 assign event test {suffix}\nbody",
        "source_url": f"{_URL_PREFIX}{suffix}",
        "is_valid": True,
        "trust_score": 0.7,
        "summary": summary,
        "extracted_info": {"embedding": embedding},
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": CategoryType.ALPHA,
        "published_at": published_at,
        "entity_names": list(names),
        "entity_names_extracted": [],
    }


async def _process(session, state: dict) -> dict:
    result = await _assign_then_save(state, session=session)
    await session.commit()
    return result


async def _event_ids_for(session, item_ids: list[uuid.UUID]) -> set[uuid.UUID]:
    rows = await session.execute(
        select(EventDocument.event_id).where(EventDocument.tech_item_id.in_(item_ids))
    )
    return {row[0] for row in rows.all()}


@pytest.mark.asyncio
async def test_a_lone_document_still_gets_an_event(session_factory):
    async with session_factory() as session:
        result = await _process(
            session,
            _state(
                "lone",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-266-test-acme",),
            ),
        )
        item_id = result["saved_item_id"]
        assert item_id is not None

    async with session_factory() as session:
        events = await _event_ids_for(session, [item_id])
        assert len(events) == 1

        item_count = await session.execute(
            select(TechItem.id).where(TechItem.id == item_id)
        )
        assert item_count.scalar_one_or_none() == item_id


@pytest.mark.asyncio
async def test_a_second_article_on_the_same_event_does_not_create_one(session_factory):
    async with session_factory() as session:
        first = await _process(
            session,
            _state(
                "dup-a",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-266-test-acme",),
            ),
        )
        second = await _process(
            session,
            _state(
                "dup-b",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-266-test-acme",),
            ),
        )

    async with session_factory() as session:
        events = await _event_ids_for(
            session, [first["saved_item_id"], second["saved_item_id"]]
        )
        # 사건 수는 그대로(1), 근거(event_documents) 수만 는다(2).
        assert len(events) == 1


@pytest.mark.asyncio
async def test_unrelated_articles_get_their_own_events(session_factory):
    async with session_factory() as session:
        a = await _process(
            session,
            _state(
                "unrelated-a",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-266-test-acme",),
            ),
        )
        b = await _process(
            session,
            _state(
                "unrelated-b",
                embedding=_embedding(0.0, 1.0),
                published_at=NOW,
                names=("arg-266-test-globex",),
            ),
        )

    async with session_factory() as session:
        events_a = await _event_ids_for(session, [a["saved_item_id"]])
        events_b = await _event_ids_for(session, [b["saved_item_id"]])
        assert events_a != events_b


@pytest.mark.asyncio
async def test_disjoint_names_keep_similar_articles_apart_on_an_empty_db(session_factory):
    """임베딩은 어느 정도 비슷하지만(코사인 0.5) 이름이 전혀 겹치지 않으면
    묶이지 않는다: 0.55*0.5 + 0.15*1.0(같은 시각) = 0.425 < join_threshold(0.55)."""
    async with session_factory() as session:
        a = await _process(
            session,
            _state(
                "disjoint-a",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-266-test-acme",),
            ),
        )
        b = await _process(
            session,
            _state(
                "disjoint-b",
                embedding=_embedding(0.5, 0.8660254037844387),  # cos(60deg) == 0.5
                published_at=NOW,
                names=("arg-266-test-globex",),
            ),
        )

    async with session_factory() as session:
        events_a = await _event_ids_for(session, [a["saved_item_id"]])
        events_b = await _event_ids_for(session, [b["saved_item_id"]])
        assert events_a != events_b


@pytest.mark.asyncio
async def test_every_assigned_document_has_an_event(session_factory):
    async with session_factory() as session:
        docs = [
            await _process(
                session,
                _state(
                    "coverage-a",
                    embedding=_embedding(1.0, 0.0),
                    published_at=NOW,
                    names=("arg-266-test-acme",),
                ),
            ),
            await _process(
                session,
                _state(
                    "coverage-b",
                    embedding=_embedding(1.0, 0.0),
                    published_at=NOW,
                    names=("arg-266-test-acme",),
                ),
            ),
            await _process(
                session,
                _state(
                    "coverage-c",
                    embedding=_embedding(0.0, 1.0),
                    published_at=NOW,
                    names=("arg-266-test-globex",),
                ),
            ),
        ]
        item_ids = [d["saved_item_id"] for d in docs]

    async with session_factory() as session:
        linked = await _event_ids_for(session, item_ids)
        rows = await session.execute(
            select(EventDocument.tech_item_id).where(
                EventDocument.tech_item_id.in_(item_ids)
            )
        )
        linked_item_ids = {row[0] for row in rows.all()}
        assert linked_item_ids == set(item_ids)
        assert len(linked) >= 1


def _embedding_on_axes(dim0: int, dim1: int, x: float, y: float) -> list[float]:
    """지정한 두 성분 위치에만 값을 채운 768차원 임베딩.

    서로 다른 성분 위치를 쓰는 두 벡터는 항상 코사인 0이다 — "실행"마다
    다른 축을 쓰면 두 실행이 서로의 후보 조회에 전혀 끼어들지 않아, 공유
    DB 위에서도 "각자 새 DB에서 시작한 것"과 같은 효과를 낸다.
    """
    vec = [0.0] * 768
    vec[dim0] = x
    vec[dim1] = y
    return vec


@pytest.mark.asyncio
async def test_the_same_input_lands_in_the_same_event_twice(session_factory):
    """서로 다른(직교하는) 임베딩 축을 두 "실행"에 나눠 써서 사실상 각자
    새 DB에서 시작한 것처럼 만든다. 같은 상대적 입력 패턴(A와 B는 묶이고
    C는 떨어진다)이 두 실행 모두에서 같은 그룹 구조로 나와야 한다."""

    async def _run(dim0: int, dim1: int, tag: str):
        async with session_factory() as session:
            a = await _process(
                session,
                _state(
                    f"det-{tag}-a",
                    embedding=_embedding_on_axes(dim0, dim1, 1.0, 0.0),
                    published_at=NOW,
                    names=(f"arg-266-test-{tag}-acme",),
                ),
            )
            b = await _process(
                session,
                _state(
                    f"det-{tag}-b",
                    embedding=_embedding_on_axes(dim0, dim1, 1.0, 0.0),
                    published_at=NOW,
                    names=(f"arg-266-test-{tag}-acme",),
                ),
            )
            c = await _process(
                session,
                _state(
                    f"det-{tag}-c",
                    embedding=_embedding_on_axes(dim0, dim1, 0.0, 1.0),
                    published_at=NOW,
                    names=(f"arg-266-test-{tag}-globex",),
                ),
            )
        async with session_factory() as session:
            event_a = await _event_ids_for(session, [a["saved_item_id"]])
            event_b = await _event_ids_for(session, [b["saved_item_id"]])
            event_c = await _event_ids_for(session, [c["saved_item_id"]])
        return event_a, event_b, event_c

    run1_a, run1_b, run1_c = await _run(0, 1, "r1")
    run2_a, run2_b, run2_c = await _run(2, 3, "r2")

    # 두 실행 모두: A와 B는 같은 사건, C는 다른 사건.
    assert run1_a == run1_b
    assert run1_a != run1_c
    assert run2_a == run2_b
    assert run2_a != run2_c


@pytest.mark.asyncio
async def test_a_lower_threshold_merges_what_a_higher_one_splits(session_factory, monkeypatch):
    """cos(theta)=0.6, 같은 시각(time decay=1.0) => score = 0.55*0.6 + 0.15*1.0 = 0.48.
    기본 join_threshold(0.55) 밑이라 갈라지고, 0.4로 낮추면 넘어서 합쳐진다.

    high/low 두 하위 실행이 같은 세션 팩토리 스코프 안에 함께 있으므로,
    같은 축을 재사용하면 low 실행의 문서가 high 실행이 이미 만들어 둔
    사건까지 후보로 봐서 오염된다 — 서로 다른 임베딩 성분 위치를 써서
    두 실행을 완전히 분리한다 (``test_the_same_input_lands_in_the_same_event_twice``
    와 같은 수법)."""
    embedding_high_a = _embedding_on_axes(4, 5, 1.0, 0.0)
    embedding_high_b = _embedding_on_axes(4, 5, 0.6, 0.8)  # cos(theta) == 0.6
    embedding_low_a = _embedding_on_axes(6, 7, 1.0, 0.0)
    embedding_low_b = _embedding_on_axes(6, 7, 0.6, 0.8)  # cos(theta) == 0.6

    async with session_factory() as session:
        high_a = await _process(
            session,
            _state(
                "threshold-high-a",
                embedding=embedding_high_a,
                published_at=NOW,
                names=("arg-266-test-threshold-high-a",),
            ),
        )
        high_b = await _process(
            session,
            _state(
                "threshold-high-b",
                embedding=embedding_high_b,
                published_at=NOW,
                names=("arg-266-test-threshold-high-b",),
            ),
        )

    async with session_factory() as session:
        events_high = await _event_ids_for(
            session, [high_a["saved_item_id"], high_b["saved_item_id"]]
        )
        assert len(events_high) == 2  # default threshold: stays split

    monkeypatch.setattr(settings.user.event_detection, "join_threshold", 0.4)
    async with session_factory() as session:
        low_a = await _process(
            session,
            _state(
                "threshold-low-a",
                embedding=embedding_low_a,
                published_at=NOW,
                names=("arg-266-test-threshold-low-a",),
            ),
        )
        low_b = await _process(
            session,
            _state(
                "threshold-low-b",
                embedding=embedding_low_b,
                published_at=NOW,
                names=("arg-266-test-threshold-low-b",),
            ),
        )

    async with session_factory() as session:
        events_low = await _event_ids_for(
            session, [low_a["saved_item_id"], low_b["saved_item_id"]]
        )
        assert len(events_low) == 1  # lowered threshold: merges


@pytest.mark.asyncio
async def test_a_failing_assignment_still_saves_the_document(session_factory, monkeypatch):
    """assign_event_node가 던지도록 갈아 끼워도 _assign_then_save는 삼키고
    save_node를 정상 호출한다 — 문서는 저장된다(부모 AC). save_node는
    event_id=None을 "새 사건이 필요하다"는 신호로도 함께 쓰므로, 이 경우도
    새 사건이 하나 생겨 붙는다 — "배정 성공한 문서 중 무소속이 없다"(다른
    부모 AC)는 이 경로에서도 깨지지 않는다."""

    async def _raise(state, *, session):
        raise RuntimeError("assign_event_node exploded")

    monkeypatch.setattr(brain_pipeline, "assign_event_node", _raise)

    async with session_factory() as session:
        result = await _process(
            session,
            _state(
                "assign-fails",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-266-test-acme",),
            ),
        )

    assert result["saved"] is True
    item_id = result["saved_item_id"]
    assert item_id is not None

    async with session_factory() as session:
        events = await _event_ids_for(session, [item_id])
        assert len(events) == 1
