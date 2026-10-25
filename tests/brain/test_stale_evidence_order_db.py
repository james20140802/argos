"""``--rename-stale``의 근거 표본 정렬 — ARG-274 DB 통합 테스트.

``generate_event_naming``은 근거를 앞에서부터 ``event_summary.max_docs``개만 쓴다.
그래서 ``fetch_stale_events``가 사건 안의 근거를 어떤 순서로 돌려주느냐가 곧
"무엇을 보고 이름을 짓느냐"다. 근거가 캡을 넘는 사건에서 방금 붙어
``naming_stale``을 세운 문서가 표본 밖으로 밀리면, 옛 근거로 지은 이름을 쓰면서
플래그만 내려 재명명이 의미를 잃는다.

패턴은 ``tests/brain/test_event_evidence_db.py``와 같다: 모듈 스코프 skip +
이 파일이 만든 행만 정리.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.event_backfill import fetch_stale_events
from argos.config import settings
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

MAX_EVIDENCE_DOCS = settings.user.event_summary.max_docs

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-274-stale-order-test.example.com/"

NOW = datetime.now(timezone.utc)

_DEFAULT = object()
"""``_item``에서 "요약을 명시하지 않음"과 "요약이 NULL"을 가르는 센티널."""


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-274 stale-evidence-order "
            "DB integration test (start the Docker DB to run it)"
        )



@pytest.fixture
async def session_factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with factory() as session:
            ids = [
                row[0]
                for row in (
                    await session.execute(
                        select(TechItem.id).where(
                            TechItem.source_url.like(f"{_URL_PREFIX}%")
                        )
                    )
                ).all()
            ]
            event_ids: set[uuid.UUID] = set()
            if ids:
                event_ids = {
                    row[0]
                    for row in (
                        await session.execute(
                            select(EventDocument.event_id).where(
                                EventDocument.tech_item_id.in_(ids)
                            )
                        )
                    ).all()
                }
                await session.execute(
                    delete(EventDocument).where(EventDocument.tech_item_id.in_(ids))
                )
            await session.execute(
                delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
            )
            if event_ids:
                await session.execute(delete(TechEvent).where(TechEvent.id.in_(event_ids)))
            await session.commit()
        await engine.dispose()


def _item(
    suffix: str,
    *,
    title: str,
    published_at: datetime,
    summary: str | None = _DEFAULT,
    digest: str | None = None,
) -> TechItem:
    return TechItem(
        id=uuid.uuid4(),
        title=title,
        source_url=f"{_URL_PREFIX}{suffix}",
        raw_content=f"body {suffix}",
        summary=f"summary {suffix}" if summary is _DEFAULT else summary,
        digest=digest,
        category=CategoryType.MAINSTREAM,
        published_at=published_at,
    )


@pytest.mark.asyncio
async def test_newest_link_survives_the_evidence_cap(session_factory):
    """근거가 캡을 넘어도 가장 최근에 붙은 문서가 표본 맨 앞에 온다.

    캡보다 두 건 많은 문서를 먼저 매달아 커밋하고(같은 트랜잭션이라 링크
    시각이 같다), 그 뒤 별도 트랜잭션으로 한 건을 더 매단다 — 온라인
    파이프라인이 나중에 문서를 붙여 naming_stale을 세우는 상황이다.
    """
    event_id = uuid.uuid4()
    old_count = MAX_EVIDENCE_DOCS + 2

    async with session_factory() as session:
        session.add(
            TechEvent(id=event_id, occurred_at=NOW, naming_stale=True, title=None)
        )
        await session.flush()
        items = [
            _item(f"old-{n}", title=f"옛 기사 {n}", published_at=NOW - timedelta(days=30))
            for n in range(old_count)
        ]
        session.add_all(items)
        await session.flush()
        session.add_all(
            [
                EventDocument(id=uuid.uuid4(), event_id=event_id, tech_item_id=i.id)
                for i in items
            ]
        )
        await session.commit()

    async with session_factory() as session:
        newest = _item("newest", title="방금 붙은 기사", published_at=NOW)
        session.add(newest)
        await session.flush()
        session.add(
            EventDocument(id=uuid.uuid4(), event_id=event_id, tech_item_id=newest.id)
        )
        await session.commit()

    async with session_factory() as session:
        stale = await fetch_stale_events(session)

    ours = [e for e in stale if e.event_id == event_id]
    assert len(ours) == 1
    docs = ours[0].docs
    assert len(docs) == old_count + 1
    # 프롬프트에 실제로 들어가는 표본 안에 있어야 한다.
    sample = [d.title for d in docs[:MAX_EVIDENCE_DOCS]]
    assert "방금 붙은 기사" in sample
    assert sample[0] == "방금 붙은 기사"


@pytest.mark.asyncio
async def test_same_batch_links_fall_back_to_article_recency(session_factory):
    """한 트랜잭션에서 매단 링크끼리는 최신 기사부터 온다 (백필 경로)."""
    event_id = uuid.uuid4()

    async with session_factory() as session:
        session.add(
            TechEvent(id=event_id, occurred_at=NOW, naming_stale=True, title=None)
        )
        await session.flush()
        items = [
            _item(f"batch-{n}", title=f"기사 {n}", published_at=NOW - timedelta(days=n))
            for n in range(3)
        ]
        session.add_all(items)
        await session.flush()
        session.add_all(
            [
                EventDocument(id=uuid.uuid4(), event_id=event_id, tech_item_id=i.id)
                for i in items
            ]
        )
        await session.commit()

    async with session_factory() as session:
        stale = await fetch_stale_events(session)

    ours = [e for e in stale if e.event_id == event_id]
    assert len(ours) == 1
    assert [d.title for d in ours[0].docs] == ["기사 0", "기사 1", "기사 2"]


@pytest.mark.asyncio
async def test_snapshot_carries_the_event_row_version(session_factory):
    """ARG-274 가드 전제 — 스냅샷이 사건 행 버전을 들고 나온다."""
    event_id = uuid.uuid4()

    async with session_factory() as session:
        session.add(
            TechEvent(id=event_id, occurred_at=NOW, naming_stale=True, title=None)
        )
        await session.flush()
        item = _item("version", title="기사", published_at=NOW)
        session.add(item)
        await session.flush()
        session.add(
            EventDocument(id=uuid.uuid4(), event_id=event_id, tech_item_id=item.id)
        )
        await session.commit()

    async with session_factory() as session:
        stale = await fetch_stale_events(session)
        row = (
            await session.execute(
                select(TechEvent.updated_at).where(TechEvent.id == event_id)
            )
        ).scalar_one()

    ours = [e for e in stale if e.event_id == event_id]
    assert len(ours) == 1
    assert ours[0].updated_at == row


@pytest.mark.asyncio
async def test_evidence_is_the_body_source_and_names_never_the_item_summary(session_factory):
    """ARG-308: 근거는 본문·출처·뽑힌 이름이다. 기사별 요약·digest는 싣지 않는다."""
    from argos.models.document_entity import DocumentEntity
    from argos.models.entity import Entity

    event_id = uuid.uuid4()
    entity = Entity(id=uuid.uuid4(), name="Claude Duet", normalized_key=f"arg308-{uuid.uuid4()}")
    try:
        async with session_factory() as session:
            session.add(TechEvent(id=event_id, occurred_at=NOW, naming_stale=True, title=None))
            session.add(entity)
            await session.flush()
            item = _item(
                "arg308-body", title="기사 제목", published_at=NOW,
                summary="ITEM SUMMARY", digest="ITEM DIGEST",
            )
            session.add(item)
            await session.flush()
            session.add(EventDocument(id=uuid.uuid4(), event_id=event_id, tech_item_id=item.id))
            session.add(DocumentEntity(tech_item_id=item.id, entity_id=entity.id))
            await session.commit()

        async with session_factory() as session:
            stale = await fetch_stale_events(session)

        (ours,) = [e for e in stale if e.event_id == event_id]
        (doc,) = ours.docs
        assert doc.summary is None
        assert doc.body == "body arg308-body"
        assert doc.source == f"{_URL_PREFIX}arg308-body"
        assert doc.names == ("Claude Duet",)
    finally:
        async with session_factory() as session:
            await session.execute(delete(Entity).where(Entity.id == entity.id))
            await session.commit()


@pytest.mark.asyncio
async def test_resummarize_targets_events_without_sections_but_not_tombstones(session_factory):
    """ARG-308: 소제목 상세가 없는 사건은 이름이 멀쩡해도 대상, 이미 있으면 제외."""
    fresh, done, tomb = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with session_factory() as session:
        session.add_all([
            TechEvent(id=fresh, occurred_at=NOW, naming_stale=False, title="옛 요약"),
            TechEvent(
                id=done, occurred_at=NOW, naming_stale=False, title="새 요약",
                event_type="Launch", sections=[{"heading": "무엇이 새로운가", "body": "b"}],
            ),
        ])
        await session.flush()
        session.add(TechEvent(id=tomb, occurred_at=NOW, title=None, merged_into_id=fresh))
        await session.flush()
        for suffix, event_id in (("fresh", fresh), ("done", done), ("tomb", tomb)):
            item = _item(f"arg308-{suffix}", title=suffix, published_at=NOW)
            session.add(item)
            await session.flush()
            session.add(EventDocument(id=uuid.uuid4(), event_id=event_id, tech_item_id=item.id))
        await session.commit()

    async with session_factory() as session:
        targets = {e.event_id for e in await fetch_stale_events(session, include_unsectioned=True)}
        renames = {e.event_id for e in await fetch_stale_events(session)}
        # tombstone은 fixture 정리 순서(사건 삭제) 전에 풀어 둔다 — RESTRICT 때문.
        await session.execute(
            TechEvent.__table__.update().where(TechEvent.id == tomb).values(merged_into_id=None)
        )
        await session.commit()

    assert fresh in targets
    assert done not in targets and tomb not in targets
    assert fresh not in renames  # --rename-stale 대상(이름 낡음/무명)은 그대로
