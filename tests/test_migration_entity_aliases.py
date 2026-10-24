"""ARG-312 DB integration test — entity_aliases migration round trip.

``alembic upgrade head`` -> ``downgrade`` (이 리비전의 부모로) -> ``upgrade head``가
기존 행을 잃지 않고, 제약(자기 별칭 금지, 별칭 이름당 한 행, 별칭 행만 지우는
CASCADE)이 DB에서 실제로 걸리는지 확인한다.

구조와 안전장치는 ``tests/test_migration_event_layer.py``를 따른다: 실행마다 새
무작위 이름의 임시 DB(``argos_migration_test_<random>``)에서만 돌고 자기가 만든
DB만 드롭하며, Postgres가 없으면(릴리즈 CI) 통째로 skip한다. dev DB(``argos``)와
pytest 스크래치 DB에는 DDL을 실행하지 않는다.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from sqlalchemy.engine.url import make_url

from argos.config import settings
from tests.conftest import DEV_DB_NAME, TEST_DB_NAME
from tests.conftest import db_reachable as _db_reachable

_BASE_URL: str = settings.database_url
_MIGRATION_DB_NAME = f"argos_migration_test_{uuid.uuid4().hex[:12]}"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALIAS_REVISION = "c7d93a5e1f48"


def _alias_down_revision() -> str:
    """이 리비전의 부모. ARG-311 뒤로 rebase하면 down_revision이 바뀌므로 읽어서 쓴다."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    revision = ScriptDirectory.from_config(cfg).get_revision(_ALIAS_REVISION)
    assert revision is not None and isinstance(revision.down_revision, str)
    return revision.down_revision


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_BASE_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-312 entity_aliases "
            "migration round-trip test (start the Docker DB to run it)"
        )
    if _MIGRATION_DB_NAME in (DEV_DB_NAME, TEST_DB_NAME):
        raise RuntimeError(
            f"migration test DB name {_MIGRATION_DB_NAME!r} collides with the dev/scratch DB"
        )


def _connect_params() -> dict:
    parsed = make_url(_BASE_URL)
    return {
        "host": parsed.host or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
    }


async def _create_migration_db() -> None:
    import asyncpg

    params = _connect_params()
    admin = await asyncpg.connect(database="postgres", **params)
    try:
        await admin.execute(f'CREATE DATABASE "{_MIGRATION_DB_NAME}"')
    finally:
        await admin.close()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    finally:
        await conn.close()


async def _drop_migration_db() -> None:
    import asyncpg

    admin = await asyncpg.connect(database="postgres", **_connect_params())
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{_MIGRATION_DB_NAME}" WITH (FORCE)')
    finally:
        await admin.close()


async def _scalar(sql: str, *args):
    import asyncpg

    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **_connect_params())
    try:
        return await conn.fetchval(sql, *args)
    finally:
        await conn.close()


async def _table_exists() -> bool:
    return await _scalar(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'entity_aliases')"
    )


async def _enum_exists() -> bool:
    return await _scalar("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alias_rule')")


async def _seed_entities() -> tuple[uuid.UUID, uuid.UUID]:
    import asyncpg

    canonical, alias = uuid.uuid4(), uuid.uuid4()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **_connect_params())
    try:
        await conn.execute(
            "INSERT INTO entities (id, name, normalized_key) VALUES "
            "($1, 'Codex', 'codex'), ($2, 'OpenAI Codex', 'openai codex')",
            canonical,
            alias,
        )
    finally:
        await conn.close()
    return canonical, alias


async def _exercise_constraints(canonical: uuid.UUID, alias: uuid.UUID) -> None:
    import asyncpg

    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **_connect_params())
    try:
        insert = (
            "INSERT INTO entity_aliases (id, alias_entity_id, canonical_entity_id, rule, evidence) "
            "VALUES (uuid_generate_v4(), $1, $2, 'VendorPrefix', '{}'::jsonb)"
        )
        await conn.execute(insert, alias, canonical)
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(insert, canonical, canonical)
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(insert, alias, canonical)
        # 대표 이름이 지워져도 별칭 이름의 entities 행은 남고 별칭 행만 사라진다.
        await conn.execute("DELETE FROM entities WHERE id = $1", canonical)
        assert await conn.fetchval("SELECT count(*) FROM entity_aliases") == 0
        assert await conn.fetchval("SELECT count(*) FROM entities WHERE id = $1", alias) == 1
    finally:
        await conn.close()


def _run_alembic(fn_name: str, revision: str) -> None:
    """임시 DB에만 alembic을 돌린다 (``test_migration_event_layer._run_alembic`` 참고)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    original_db = settings.secrets.POSTGRES_DB
    settings.secrets.POSTGRES_DB = _MIGRATION_DB_NAME
    try:
        getattr(command, fn_name)(cfg, revision)
    finally:
        settings.secrets.POSTGRES_DB = original_db


def test_entity_aliases_migration_round_trip():
    # alembic이 내부에서 asyncio.run()을 쓰므로 동기 테스트로 둔다.
    asyncio.run(_create_migration_db())
    try:
        _run_alembic("upgrade", "head")
        assert asyncio.run(_table_exists()) is True
        assert asyncio.run(_enum_exists()) is True

        canonical, alias = asyncio.run(_seed_entities())
        asyncio.run(_exercise_constraints(canonical, alias))

        _run_alembic("downgrade", _alias_down_revision())
        assert asyncio.run(_table_exists()) is False
        assert asyncio.run(_enum_exists()) is False
        # 기존 행 보존: 대표 이름은 위에서 지웠고, 별칭 이름의 entities 행은 남아 있어야 한다.
        assert asyncio.run(_scalar("SELECT count(*) FROM entities WHERE id = $1", alias)) == 1

        _run_alembic("upgrade", "head")
        assert asyncio.run(_table_exists()) is True
        assert asyncio.run(_enum_exists()) is True
    finally:
        asyncio.run(_drop_migration_db())
