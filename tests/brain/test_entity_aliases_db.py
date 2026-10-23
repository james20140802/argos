"""별칭 층의 DB 통합 테스트 (ARG-312).

- ``plan_aliases`` -> ``apply_aliases`` -> 다시 ``plan_aliases``가 멱등이다.
- ``Manual`` 행은 자동 재계산이 건드리지 않는다.
- **제약 회귀(T8):** 별칭 행을 넣기 전후로 사건 묶기가 읽는 이름 신호
  (``raw_names_for_documents`` / ``names_for_documents`` / ``filter_common_names``)가
  그대로다. 별칭 층은 opt-in이고 저장된 이름 경로를 바꾸지 않는다.

패턴은 ``test_entity_store_db.py``와 같다. 이 모듈의 키는 전부 ``zzarg`` 접두어를
달고, 끝나면 그 행만 지운다. Postgres가 없으면 통째로 skip한다.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain import alias_store
from argos.brain.entity_aliases import AliasRule
from argos.brain.entity_store import (
    filter_common_names,
    names_for_documents,
    raw_names_for_documents,
)
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.entity_alias import EntityAlias
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-312-entity-alias-test.example.com/"
_KEY_LIKE = "zzarg%"

_VENDOR = "zzargvend"
_BASE = "zzargcodex"
_ALIAS = f"{_VENDOR} {_BASE}"


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-312 entity alias DB "
            "integration test (start the Docker DB to run it)"
        )


@pytest.fixture
async def session_factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with factory() as session:
            await session.execute(
                delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
            )
            # 별칭 행은 FK CASCADE로 같이 지워진다 — 이 규칙이 실제로 도는지의 증거이기도 하다.
            await session.execute(delete(Entity).where(Entity.normalized_key.like(_KEY_LIKE)))
            await session.commit()
        await engine.dispose()


async def _seed(session) -> dict[str, uuid.UUID]:
    """뿌리 하나(따르는 이름 3개), 접두어 이름과 그 대표, 공백 변형 한 쌍, 문서 3건."""
    entity_ids: dict[str, uuid.UUID] = {}
    keys = [
        _VENDOR, _ALIAS, _BASE,
        f"{_VENDOR} zzargalpha", f"{_VENDOR} zzargbeta", "zzargalpha", "zzargbeta",
        "zzarghug face", "zzarghugface",
        "zzargman one", "zzargman two",
    ]
    for key in keys:
        entity_ids[key] = uuid.uuid4()
        session.add(Entity(id=entity_ids[key], name=key.title(), normalized_key=key))
    await session.flush()

    item_ids = []
    for i in range(3):
        item = TechItem(
            id=uuid.uuid4(),
            title=f"ARG-312 alias test {i}",
            source_url=f"{_URL_PREFIX}{i}",
            raw_content="Zzargcodex shipped today. ZzargVend Zzargcodex is great.",
            category=CategoryType.ALPHA,
        )
        session.add(item)
        item_ids.append(item.id)
    await session.flush()
    links = {
        _ALIAS: item_ids,  # 접두어 표기가 3개 문서에 그대로 있다
        _BASE: item_ids[:1],
        _VENDOR: item_ids,
        "zzarghug face": item_ids[:1],
        "zzarghugface": item_ids[:2],
    }
    for key, items in links.items():
        for item_id in items:
            session.add(
                DocumentEntity(id=uuid.uuid4(), tech_item_id=item_id, entity_id=entity_ids[key])
            )
    await session.flush()
    entity_ids["__items__"] = item_ids  # type: ignore[assignment]
    return entity_ids


def _mine(links):
    return {(link.alias_key, link.canonical_key) for link in links if link.alias_key.startswith("zzarg")}


async def test_plan_apply_plan_is_idempotent_and_leaves_manual_rows(session_factory):
    async with session_factory() as session:
        ids = await _seed(session)
        session.add(
            EntityAlias(
                alias_entity_id=ids["zzargman one"],
                canonical_entity_id=ids["zzargman two"],
                rule=AliasRule.MANUAL,
                evidence={},
            )
        )
        await session.commit()

    async with session_factory() as session:
        plan = await alias_store.plan_aliases(session)
        assert _mine(plan.added) == {
            ("zzarghug face", "zzarghugface"),
            # 문서가 3건으로 더 많은 접두어 표기가 대표가 된다 (표시 이름은 읽기 시점에 정한다)
            (_BASE, _ALIAS),
        }
        assert not {k for pair in _mine(plan.removed) for k in pair}
        changed = await alias_store.apply_aliases(session, plan)
        assert changed >= 2
        await session.commit()

    async with session_factory() as session:
        again = await alias_store.plan_aliases(session)
        assert _mine(again.added) == set() and _mine(again.removed) == set()
        rows = (
            await session.execute(
                select(EntityAlias.alias_entity_id, EntityAlias.rule).where(
                    EntityAlias.alias_entity_id.in_(
                        [ids["zzargman one"], ids["zzarghug face"], ids[_BASE]]
                    )
                )
            )
        ).all()
        by_alias = {alias_id: rule for alias_id, rule in rows}
        assert by_alias[ids["zzargman one"]] is AliasRule.MANUAL
        assert by_alias[ids["zzarghug face"]] is AliasRule.SPACING
        assert by_alias[ids[_BASE]] is AliasRule.VENDOR_PREFIX


async def test_aliases_do_not_change_the_name_signals_event_grouping_reads(session_factory):
    async with session_factory() as session:
        ids = await _seed(session)
        await session.commit()
    item_ids = ids["__items__"]

    async def snapshot(session):
        keys = frozenset({_ALIAS, _BASE, _VENDOR, "zzarghug face", "zzarghugface"})
        return (
            await raw_names_for_documents(session, item_ids),
            await names_for_documents(session, item_ids),
            await filter_common_names(session, keys),
            (await session.execute(select(Entity.id, Entity.normalized_key, Entity.name)))
            .all(),
            (await session.execute(select(DocumentEntity.tech_item_id, DocumentEntity.entity_id)))
            .all(),
        )

    async with session_factory() as session:
        before = await snapshot(session)
        plan = await alias_store.plan_aliases(session)
        # 별칭이 실제로 만들어지는 상태에서 비교한다
        assert {(_BASE, _ALIAS), ("zzarghug face", "zzarghugface")} <= _mine(plan.added)
        await alias_store.apply_aliases(session, plan)
        await session.commit()

    async with session_factory() as session:
        after = await snapshot(session)
    assert sorted(map(str, before[0].items())) == sorted(map(str, after[0].items()))
    assert before[1] == after[1]
    assert before[2] == after[2]
    assert sorted(map(tuple, before[3])) == sorted(map(tuple, after[3]))
    assert sorted(map(tuple, before[4])) == sorted(map(tuple, after[4]))
