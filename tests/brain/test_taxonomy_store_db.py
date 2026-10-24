"""taxonomy_store apply 왕복의 DB 통합 테스트 (ARG-311 T4).

``tests/brain/test_entity_store_db.py``와 같은 패턴: 스크래치 DB(argos_test*)만
쓰고, Postgres가 없으면 통째로 skip(릴리즈 CI 대비).
"""
from __future__ import annotations

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain import taxonomy as tx
from argos.brain.taxonomy_store import apply_plan, load_confirmed, plan_apply
from argos.config import settings
from argos.models.role_branch import RoleBranch
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping ARG-311 role_branches DB test")


@pytest.fixture
async def session_factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(RoleBranch))
        await session.commit()
    try:
        yield factory
    finally:
        async with factory() as session:
            await session.execute(delete(RoleBranch))
            await session.commit()
        await engine.dispose()


async def _apply(factory, new):
    async with factory() as session:
        current = await load_confirmed(session)
        plan = plan_apply(current, new)
        await apply_plan(session, plan)
        await session.commit()
        return plan


@pytest.mark.asyncio
async def test_seed_round_trip_and_ids_survive_rename(session_factory):
    await _apply(session_factory, tx.SPIKE_SEED)
    async with session_factory() as session:
        loaded = await load_confirmed(session)
    assert loaded is not None
    assert [b.name for b in loaded] == [b.name for b in tx.SPIKE_SEED]
    assert [[c.name for c in b.children] for b in loaded] == [
        [c.name for c in b.children] for b in tx.SPIKE_SEED
    ]
    assert tx.validate(loaded) == []

    target = "벤치마크 (Benchmarks)"
    old_id = tx.find(loaded, target)[1].id
    assert old_id
    await _apply(session_factory, tx.rename(loaded, target, "벤치마크"))
    async with session_factory() as session:
        again = await load_confirmed(session)
    assert tx.find(again, "벤치마크")[1].id == old_id


@pytest.mark.asyncio
async def test_unchanged_apply_is_idempotent(session_factory):
    await _apply(session_factory, tx.SPIKE_SEED)
    async with session_factory() as session:
        loaded = await load_confirmed(session)
        plan = plan_apply(loaded, loaded)
    assert plan.is_empty


@pytest.mark.asyncio
async def test_merge_split_and_name_swap_apply_cleanly(session_factory):
    await _apply(session_factory, tx.SPIKE_SEED)
    async with session_factory() as session:
        tree = await load_confirmed(session)

    a, b = "버전 관리·협업 (Version Control)", "런타임·배포 (Runtime & Deployment)"
    # 이름 맞바꾸기 — 유일 제약 때문에 임시 이름 단계가 필요하다.
    swapped = tx.rename(tx.rename(tree, a, "임시 이름"), b, a)
    swapped = tx.rename(swapped, "임시 이름", b)
    await _apply(session_factory, swapped)

    async with session_factory() as session:
        tree = await load_confirmed(session)
    merged = tx.merge(tree, "평가", "AI 모델 및 엔진")
    await _apply(session_factory, merged)
    async with session_factory() as session:
        final = await load_confirmed(session)
        count = await session.scalar(select(func.count()).select_from(RoleBranch))
    assert "평가" not in [t.name for t in final]
    assert count == 4 + 11  # 위 갈래 하나가 흡수돼도 아래 갈래 11개는 그대로
    assert tx.find(final, "벤치마크 (Benchmarks)")[2].name == "AI 모델 및 엔진"


@pytest.mark.asyncio
async def test_unique_name_constraint_holds(session_factory):
    await _apply(session_factory, tx.SPIKE_SEED)
    async with session_factory() as session:
        session.add(RoleBranch(name="평가", description="중복 이름 시도는 거부되어야 한다."))
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()
