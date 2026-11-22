"""ARG-333 DB integration test — event_deep_analyses migration round trip.

``alembic upgrade head`` -> ``downgrade`` (이 리비전의 부모로) -> ``upgrade head``가
기존 사건 행을 잃지 않고, 저장 자리의 약속(사건당 한 행, 결과 칸과 진행 칸을
따로 비우기, 사건 삭제 시 함께 삭제, 상태 값 제한)이 DB에서 실제로 걸리는지 확인한다.

구조와 안전장치는 ``tests/test_migration_entity_aliases.py``를 따른다: 실행마다 새
무작위 이름의 임시 DB(``argos_migration_test_<random>``)에서만 돌고 자기가 만든
DB만 드롭하며, Postgres가 없으면(릴리즈 CI) 통째로 skip한다. dev DB(``argos``)와
pytest 스크래치 DB에는 DDL을 실행하지 않는다.
"""
from __future__ import annotations

import asyncio
import json
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
_REVISION = "a7c41e9d2b63"
_TABLE = "event_deep_analyses"
_ENUM = "deep_analysis_status"


def _down_revision() -> str:
    """이 리비전의 부모. rebase로 down_revision이 바뀔 수 있으므로 읽어서 쓴다."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    revision = ScriptDirectory.from_config(cfg).get_revision(_REVISION)
    assert revision is not None and isinstance(revision.down_revision, str)
    return revision.down_revision


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_BASE_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-333 event_deep_analyses "
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
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
        _TABLE,
    )


async def _enum_exists() -> bool:
    return await _scalar("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = $1)", _ENUM)


async def _seed_events() -> tuple[uuid.UUID, uuid.UUID]:
    import asyncpg

    kept, deleted = uuid.uuid4(), uuid.uuid4()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **_connect_params())
    try:
        await conn.execute(
            "INSERT INTO tech_events (id, title, occurred_at) VALUES "
            "($1, 'kept event', now()), ($2, 'deleted event', now())",
            kept,
            deleted,
        )
    finally:
        await conn.close()
    return kept, deleted


async def _exercise_storage(kept: uuid.UUID, deleted: uuid.UUID) -> None:
    import asyncpg

    sections = json.dumps([{"heading": "성숙도", "body": "연구 프리뷰"}])
    doc_ids = [uuid.uuid4(), uuid.uuid4()]
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **_connect_params())
    try:
        # 진행 상태만 있는 행 — 한 번도 성공한 적 없는 사건.
        await conn.execute(
            f"INSERT INTO {_TABLE} (id, event_id, status, requested_at) "
            "VALUES (uuid_generate_v4(), $1, 'Waiting', now())",
            kept,
        )
        assert await conn.fetchval(f"SELECT sections FROM {_TABLE} WHERE event_id = $1", kept) is None

        # 사건당 한 행.
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                f"INSERT INTO {_TABLE} (id, event_id) VALUES (uuid_generate_v4(), $1)", kept
            )

        # 성공: 결과를 채우고 진행 상태를 비운다.
        await conn.execute(
            f"UPDATE {_TABLE} SET sections = $2::jsonb, analyzed_at = now(), doc_count = 2, "
            "doc_ids = $3::uuid[], model = 'gemma4:26b', status = NULL, requested_at = NULL "
            "WHERE event_id = $1",
            kept,
            sections,
            doc_ids,
        )
        # 다시 분석이 실패: 진행 칸만 바꾼다 — 직전 결과는 남는다.
        await conn.execute(
            f"UPDATE {_TABLE} SET status = 'Failed', failure_reason = '시간이 초과됐습니다' "
            "WHERE event_id = $1",
            kept,
        )
        row = await conn.fetchrow(
            f"SELECT sections, doc_count, doc_ids, model, status, failure_reason "
            f"FROM {_TABLE} WHERE event_id = $1",
            kept,
        )
        assert json.loads(row["sections"]) == [{"heading": "성숙도", "body": "연구 프리뷰"}]
        assert row["doc_count"] == 2
        assert list(row["doc_ids"]) == doc_ids
        assert row["model"] == "gemma4:26b"
        assert row["status"] == "Failed"
        assert row["failure_reason"] == "시간이 초과됐습니다"

        # 상태 값은 정해진 셋뿐이다.
        with pytest.raises(asyncpg.InvalidTextRepresentationError):
            await conn.execute(f"UPDATE {_TABLE} SET status = 'Done' WHERE event_id = $1", kept)

        # 사건이 지워지면 분석 행도 함께 지워진다(다른 사건의 행은 그대로).
        await conn.execute(
            f"INSERT INTO {_TABLE} (id, event_id, status) VALUES (uuid_generate_v4(), $1, 'Running')",
            deleted,
        )
        await conn.execute("DELETE FROM tech_events WHERE id = $1", deleted)
        assert await conn.fetchval(f"SELECT count(*) FROM {_TABLE} WHERE event_id = $1", deleted) == 0
        assert await conn.fetchval(f"SELECT count(*) FROM {_TABLE} WHERE event_id = $1", kept) == 1
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


def test_event_deep_analyses_migration_round_trip():
    # alembic이 내부에서 asyncio.run()을 쓰므로 동기 테스트로 둔다.
    asyncio.run(_create_migration_db())
    try:
        _run_alembic("upgrade", "head")
        assert asyncio.run(_table_exists()) is True
        assert asyncio.run(_enum_exists()) is True

        kept, deleted = asyncio.run(_seed_events())
        asyncio.run(_exercise_storage(kept, deleted))

        _run_alembic("downgrade", _down_revision())
        assert asyncio.run(_table_exists()) is False
        assert asyncio.run(_enum_exists()) is False
        # 기존 행 보존: 내려도 사건 자체는 그대로다.
        assert asyncio.run(_scalar("SELECT count(*) FROM tech_events WHERE id = $1", kept)) == 1

        _run_alembic("upgrade", "head")
        assert asyncio.run(_table_exists()) is True
        assert asyncio.run(_enum_exists()) is True
        assert asyncio.run(_scalar(f"SELECT count(*) FROM {_TABLE}")) == 0
    finally:
        asyncio.run(_drop_migration_db())
