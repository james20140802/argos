"""SimHash 저장 + 근거 수 집계 — ARG-267 DB 통합 테스트. 부모 AC를 종단으로
확인한다.

패턴은 ``tests/brain/test_assign_event_db.py``와 같다: 모듈 스코프
``session_factory`` 픽스처(NullPool) + 이 모듈이 만든 행만 정리 + Postgres가
없으면 통째로 skip.

``argos.brain.pipeline._assign_then_save``를 그대로 써서 save_node의 실제
호출 경로를 종단으로 검증한다. state의 ``simhash``는 파이프라인의
``_attach_extracted_names`` 단계가 이미 계산해 실어 준 값을 흉내 낸다 —
``tests/brain/test_assign_event_db.py``가 ``entity_names``를 미리 채워 두는
것과 같은 관례다.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.event_evidence import evidence_count
from argos.brain.near_duplicate import simhash as compute_simhash
from argos.brain.pipeline import _assign_then_save
from argos.brain.simhash_storage import from_storage, to_storage
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-267-evidence-test.example.com/"

NOW = datetime.now(timezone.utc)


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-267 evidence-count DB "
            "integration test (start the Docker DB to run it)"
        )


@pytest.fixture
async def session_factory():
    """NullPool 기반 sessionmaker를 주고, 끝나면 이 파일이 만든 행만 지운다."""
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
                delete(Entity).where(Entity.normalized_key.like("arg-267-test-%"))
            )
            await session.commit()
        await engine.dispose()


def _embedding(x: float, y: float) -> list[float]:
    """768차원 임베딩. 앞 두 성분만 채운다."""
    vec = [0.0] * 768
    vec[0] = x
    vec[1] = y
    return vec


def _state(
    suffix: str,
    *,
    raw_text: str,
    embedding: list[float],
    published_at: datetime,
    names: tuple[str, ...] = (),
) -> dict:
    return {
        "raw_text": raw_text,
        "source_url": f"{_URL_PREFIX}{suffix}",
        "is_valid": True,
        "trust_score": 0.7,
        "summary": "",
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
        # 파이프라인의 _attach_extracted_names 단계가 이미 계산해 실어 주는
        # 값을 흉내 낸다 — 이 모듈은 save_node의 저장 경로를 검증하는 것이지
        # 파이프라인 배선(test_pipeline.py 소관)을 검증하는 게 아니다.
        "simhash": compute_simhash(raw_text),
    }


async def _process(session, state: dict) -> dict:
    result = await _assign_then_save(state, session=session)
    await session.commit()
    return result


@pytest.mark.asyncio
async def test_saving_a_document_records_its_simhash(session_factory):
    raw_text = "ARG-267 simhash storage test\nSome body content about a widget."
    async with session_factory() as session:
        result = await _process(
            session,
            _state(
                "simhash-store",
                raw_text=raw_text,
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-267-test-acme",),
            ),
        )
        item_id = result["saved_item_id"]
        assert item_id is not None

    async with session_factory() as session:
        stored = await session.execute(
            select(TechItem.simhash).where(TechItem.id == item_id)
        )
        stored_value = stored.scalar_one()
        assert stored_value is not None
        assert from_storage(stored_value) == compute_simhash(raw_text)


@pytest.mark.asyncio
async def test_a_syndicated_copy_lands_in_the_same_event(session_factory):
    """같은 본문·다른 URL 두 문서 → event_documents 링크 2행이 같은 event_id."""
    raw_text = (
        "ARG-267 syndicated copy test\nThe exact same article body, "
        "republished verbatim on a different outlet's domain."
    )
    async with session_factory() as session:
        first = await _process(
            session,
            _state(
                "syndicated-original",
                raw_text=raw_text,
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-267-test-acme",),
            ),
        )
        second = await _process(
            session,
            _state(
                "syndicated-copy",
                raw_text=raw_text,
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-267-test-acme",),
            ),
        )

    async with session_factory() as session:
        rows = await session.execute(
            select(EventDocument.event_id, EventDocument.tech_item_id).where(
                EventDocument.tech_item_id.in_(
                    [first["saved_item_id"], second["saved_item_id"]]
                )
            )
        )
        links = rows.all()
        assert len(links) == 2
        event_ids = {row[0] for row in links}
        assert len(event_ids) == 1


@pytest.mark.asyncio
async def test_the_evidence_count_stays_one_for_a_syndicated_copy(session_factory):
    raw_text = (
        "ARG-267 evidence count test\nAnother verbatim article body used to "
        "check that the derived evidence count folds the republish."
    )
    async with session_factory() as session:
        first = await _process(
            session,
            _state(
                "evidence-original",
                raw_text=raw_text,
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-267-test-acme",),
            ),
        )
        await _process(
            session,
            _state(
                "evidence-copy",
                raw_text=raw_text,
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-267-test-acme",),
            ),
        )

    async with session_factory() as session:
        event_row = await session.execute(
            select(EventDocument.event_id).where(
                EventDocument.tech_item_id == first["saved_item_id"]
            )
        )
        event_id = event_row.scalar_one()
        count = await evidence_count(session, event_id)
        assert count == 1


@pytest.mark.asyncio
async def test_two_genuinely_different_documents_count_twice(session_factory):
    """같은 사건에 억지로 매단 서로 다른 본문 2건 → evidence_count == 2."""
    async with session_factory() as session:
        first = await _process(
            session,
            _state(
                "different-a",
                raw_text="ARG-267 first genuinely distinct article about widgets.",
                embedding=_embedding(1.0, 0.0),
                published_at=NOW,
                names=("arg-267-test-acme",),
            ),
        )

    # 두 번째 문서는 첫 문서와 본문이 완전히 달라 자연 배정으로는 같은 사건에
    # 묶이지 않는다. evidence_count 자체(같은 사건에 서로 다른 SimHash를 가진
    # 두 문서가 매달렸을 때 2로 세는지)를 확인하려는 것이 이 테스트의 목적이므로,
    # 배정 로직을 거치지 않고 링크를 직접 만든다. 해시는 첫 문서 해시의
    # 비트 반전(해밍 거리 64)으로 만들어 "충분히 멀다"를 우연에 맡기지 않는다.
    async with session_factory() as session:
        first_hash = compute_simhash(
            "ARG-267 first genuinely distinct article about widgets."
        )
        far_hash = first_hash ^ ((1 << 64) - 1)
        second_item = TechItem(
            id=uuid.uuid4(),
            title="Second",
            source_url=f"{_URL_PREFIX}different-b",
            raw_content="ARG-267 second, wholly unrelated article about spacecraft.",
            category=CategoryType.ALPHA,
            simhash=to_storage(far_hash),
        )
        session.add(second_item)
        await session.flush()

        event_row = await session.execute(
            select(EventDocument.event_id).where(
                EventDocument.tech_item_id == first["saved_item_id"]
            )
        )
        event_id = event_row.scalar_one()
        await session.execute(
            EventDocument.__table__.insert().values(
                id=uuid.uuid4(), event_id=event_id, tech_item_id=second_item.id
            )
        )
        await session.commit()

        count = await evidence_count(session, event_id)
        assert count == 2


@pytest.mark.asyncio
async def test_the_count_is_not_stored_on_the_event(session_factory):
    """tech_events에 근거 수 컬럼이 없다 — 파생값이라는 계약의 회귀 방지."""
    async with session_factory() as session:
        rows = await session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'tech_events'"
            )
        )
        columns = {row[0] for row in rows.all()}
        assert not any("evidence" in c.lower() for c in columns)
