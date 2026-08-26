"""entity_store — 멱등 쓰기와 조회의 DB 통합 테스트 (ARG-263).

패턴은 ``tests/test_tech_event_tombstone_db.py``와 같다: 모듈 스코프
``session_factory`` 픽스처(NullPool) + 이 모듈이 만든 행만 정리 + Postgres가
없으면 통째로 skip.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.entity_extraction import ExtractedName
from argos.brain.entity_store import attach_names, names_for_documents
from argos.brain.graph_state import BrainState
from argos.brain.nodes.save import save_node
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-263-entity-store-test.example.com/"


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-263 entity_store DB "
            "integration test (start the Docker DB to run it)"
        )


@pytest.fixture
async def session_factory():
    """NullPool 기반 sessionmaker를 주고, 끝나면 이 파일이 만든 행을 지운다."""
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
            if item_ids:
                await session.execute(
                    delete(DocumentEntity).where(DocumentEntity.tech_item_id.in_(item_ids))
                )
            await session.execute(
                delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
            )
            await session.execute(
                delete(Entity).where(
                    Entity.normalized_key.like("arg-263-test-%")
                )
            )
            await session.commit()
        await engine.dispose()


async def _make_item(session, suffix: str) -> uuid.UUID:
    item = TechItem(
        id=uuid.uuid4(),
        title=f"ARG-263 entity store test {suffix}",
        source_url=f"{_URL_PREFIX}{suffix}",
        raw_content="x",
        category=CategoryType.ALPHA,
    )
    session.add(item)
    await session.flush()
    return item.id


def _names(*pairs: tuple[str, str]) -> list[ExtractedName]:
    return [ExtractedName(canonical=key, surface=surface) for key, surface in pairs]


@pytest.mark.asyncio
async def test_attach_names_creates_entities_and_links(session_factory):
    """문서 1건 + 이름 2개 → entities 2행, document_entities 2행."""
    async with session_factory() as session:
        item_id = await _make_item(session, "attach-basic")
        await attach_names(
            session,
            item_id,
            _names(
                ("arg-263-test-anthropic", "Anthropic"),
                ("arg-263-test-openai", "OpenAI"),
            ),
        )
        await session.commit()

    async with session_factory() as session:
        links = await names_for_documents(session, [item_id])
        assert links[item_id] == frozenset({"arg-263-test-anthropic", "arg-263-test-openai"})


@pytest.mark.asyncio
async def test_attaching_the_same_names_twice_adds_no_duplicate_links(session_factory):
    """같은 문서에 같은 이름을 두 번 attach → 링크 수가 그대로 (재크롤 멱등)."""
    async with session_factory() as session:
        item_id = await _make_item(session, "attach-idempotent")
        names = _names(("arg-263-test-anthropic", "Anthropic"))
        await attach_names(session, item_id, names)
        await session.commit()

    async with session_factory() as session:
        await attach_names(session, item_id, names)
        await session.commit()

    async with session_factory() as session:
        links = await names_for_documents(session, [item_id])
        assert links[item_id] == frozenset({"arg-263-test-anthropic"})


@pytest.mark.asyncio
async def test_the_same_name_on_two_documents_reuses_one_entity_row(session_factory):
    """같은 이름을 서로 다른 문서 둘에 attach → entities는 1행, 링크는 2행."""
    async with session_factory() as session:
        item_a = await _make_item(session, "shared-a")
        item_b = await _make_item(session, "shared-b")
        names = _names(("arg-263-test-shared-name", "Shared Name"))
        await attach_names(session, item_a, names)
        await attach_names(session, item_b, names)
        await session.commit()

    async with session_factory() as session:
        entity_rows = await session.execute(
            Entity.__table__.select().where(Entity.normalized_key == "arg-263-test-shared-name")
        )
        assert len(entity_rows.all()) == 1

        links = await names_for_documents(session, [item_a, item_b])
        assert links[item_a] == frozenset({"arg-263-test-shared-name"})
        assert links[item_b] == frozenset({"arg-263-test-shared-name"})


@pytest.mark.asyncio
async def test_names_for_documents_reads_back_what_was_attached(session_factory):
    """이름이 없는 문서는 빈 frozenset이 아니라 키 자체가 없다."""
    async with session_factory() as session:
        item_with_names = await _make_item(session, "has-names")
        item_without_names = await _make_item(session, "no-names")
        await attach_names(
            session, item_with_names, _names(("arg-263-test-lonely", "Lonely"))
        )
        await session.commit()

    async with session_factory() as session:
        links = await names_for_documents(session, [item_with_names, item_without_names])
        assert links[item_with_names] == frozenset({"arg-263-test-lonely"})
        assert item_without_names not in links
        assert links.get(item_without_names, frozenset()) == frozenset()


@pytest.mark.asyncio
async def test_save_node_attaches_the_names_carried_on_the_state(session_factory):
    """state["entity_names_extracted"]를 담고 save_node를 부르면 링크가 생긴다."""
    state: BrainState = {
        "raw_text": "ARG-263 entity store save_node test\nSome body text.",
        "source_url": f"{_URL_PREFIX}save-node",
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
        "category": CategoryType.ALPHA,
        "entity_names_extracted": _names(("arg-263-test-save-node-name", "Save Node Name")),
    }

    async with session_factory() as session:
        result = await save_node(state, session=session)
        await session.commit()
        item_id = result["saved_item_id"]

    async with session_factory() as session:
        links = await names_for_documents(session, [item_id])
        assert links[item_id] == frozenset({"arg-263-test-save-node-name"})
