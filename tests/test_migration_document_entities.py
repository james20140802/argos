"""ARG-262 DB integration test — document_entities + tech_items.simhash migration
round trip.

Verifies "``alembic upgrade head`` → ``downgrade -1`` runs without losing
existing data" for the revision that adds the ``document_entities`` link
table and the ``tech_items.simhash`` column (down_revision ``855cb67b5096``,
the event-layer revision from ARG-258):

- ``upgrade head`` creates ``document_entities`` and adds ``tech_items.simhash``.
- ``downgrade -1`` drops ``document_entities`` and the ``simhash`` column, and
  existing ``tech_items`` rows keep their id and ``source_url`` unchanged.
- ``upgrade head`` again brings both back.

Structure follows ``tests/test_migration_event_layer.py`` (ARG-258) closely:
a per-run randomized throwaway DB name, ``_assert_migration_db_is_disposable``
as a last-resort guard against dropping the dev/scratch DB, and a
``db_reachable`` self-skip so this test is a no-op in CI environments with no
Postgres service — see that module's docstring for the full rationale, which
applies unchanged here.
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

# Captured at import time, same pattern as tests/test_migration_event_layer.py.
_BASE_URL: str = settings.database_url
_MIGRATION_DB_PREFIX = "argos_migration_test"
# 실행마다 새 이름 — 같은 Postgres를 공유하는 다른 워크트리의 실행과 겹치지
# 않게. 덕분에 이 모듈은 자기가 만든 DB만 드롭한다.
_MIGRATION_DB_NAME = f"{_MIGRATION_DB_PREFIX}_{uuid.uuid4().hex[:12]}"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_migration_db_is_disposable(
    migration_db: str, dev_db: str, scratch_db: str
) -> None:
    """``migration_db``가 지워도 되는 이름인지 확인한다 — 아니면 던진다.

    tests/test_migration_event_layer.py의 동명 함수와 같은 이유로 존재한다:
    1차 방어(랜덤 접미사)가 뚫렸을 때의 마지막 방어선. skip이 아니라 예외로
    멈추는 것도 같은 이유 — 이건 파괴적 작업 직전이라 조용히 넘어가면 안 된다.
    """
    if migration_db in (dev_db, scratch_db):
        raise RuntimeError(
            f"migration test DB name {migration_db!r} collides with the "
            f"{'dev' if migration_db == dev_db else 'pytest scratch'} database "
            f"(dev={dev_db!r}, scratch={scratch_db!r}). This module runs "
            f'`DROP DATABASE IF EXISTS "{migration_db}" WITH (FORCE)` at '
            f"teardown — running it against that database would destroy it. "
            f"Point POSTGRES_DB at a different database, or restore the "
            f"per-run random suffix on the throwaway migration DB name."
        )


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_BASE_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-262 document_entities "
            "migration round-trip test (start the Docker DB to run it)"
        )
    _assert_migration_db_is_disposable(
        _MIGRATION_DB_NAME, DEV_DB_NAME, TEST_DB_NAME
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
    """Create this run's throwaway migration DB and enable its extensions.

    Plain ``CREATE DATABASE`` — deliberately **no** preceding
    ``DROP ... WITH (FORCE)``. The name carries a per-run random suffix, so if
    it somehow already exists that is a genuine surprise and should fail loudly
    rather than destroy whatever is there. Teardown only ever drops the
    database this function created.
    """
    import asyncpg

    params = _connect_params()
    admin_conn = await asyncpg.connect(database="postgres", **params)
    try:
        await admin_conn.execute(f'CREATE DATABASE "{_MIGRATION_DB_NAME}"')
    finally:
        await admin_conn.close()

    db_conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    try:
        await db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await db_conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    finally:
        await db_conn.close()


async def _drop_migration_db() -> None:
    import asyncpg

    params = _connect_params()
    admin_conn = await asyncpg.connect(database="postgres", **params)
    try:
        await admin_conn.execute(
            f'DROP DATABASE IF EXISTS "{_MIGRATION_DB_NAME}" WITH (FORCE)'
        )
    finally:
        await admin_conn.close()


async def _insert_sample_tech_items() -> list[tuple[uuid.UUID, str]]:
    """Insert two rows standing in for the dev DB's real documents."""
    import asyncpg

    params = _connect_params()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    rows: list[tuple[uuid.UUID, str]] = []
    try:
        for i in range(2):
            item_id = uuid.uuid4()
            source_url = f"https://arg262-migration-test.example/{item_id}"
            await conn.execute(
                """
                INSERT INTO tech_items
                    (id, title, source_url, raw_content, category, trust_score,
                     created_at, updated_at)
                VALUES ($1, $2, $3, $4, 'Mainstream', 0.5, now(), now())
                """,
                item_id,
                f"ARG-262 migration fixture {i}",
                source_url,
                "fixture raw content",
            )
            rows.append((item_id, source_url))
    finally:
        await conn.close()
    return rows


async def _fetch_tech_items(ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
    import asyncpg

    params = _connect_params()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    try:
        records = await conn.fetch(
            "SELECT id, source_url FROM tech_items WHERE id = ANY($1::uuid[])",
            ids,
        )
    finally:
        await conn.close()
    return {r["id"]: r["source_url"] for r in records}


async def _document_entities_table_exists() -> bool:
    import asyncpg

    params = _connect_params()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    try:
        return await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'document_entities')"
        )
    finally:
        await conn.close()


async def _simhash_column_type() -> str | None:
    import asyncpg

    params = _connect_params()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    try:
        return await conn.fetchval(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'tech_items' AND column_name = 'simhash'"
        )
    finally:
        await conn.close()


def _run_alembic(fn_name: str, revision: str) -> None:
    """Run one alembic command against the throwaway migration DB only.

    See tests/test_migration_event_layer.py::_run_alembic for the full
    rationale — same construction here: temporarily point the shared
    ``settings`` singleton's ``POSTGRES_DB`` at the throwaway DB, restore in
    ``finally``, and build the ``Config`` without ``alembic.ini`` so
    ``env.py`` never calls ``logging.config.fileConfig()`` (which would tear
    down loggers already configured elsewhere in this pytest process).
    """
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


def test_document_entities_migration_round_trip():
    """upgrade head -> downgrade -1 -> upgrade head loses no data (ARG-262).

    This is a plain (non-async) test function on purpose: alembic's
    ``command.upgrade``/``command.downgrade`` call ``asyncio.run()``
    internally (see ``alembic/env.py::run_migrations_online``), which raises
    if invoked from inside an already-running event loop. Keeping this test
    synchronous and using ``asyncio.run()`` per async helper call avoids that
    nesting entirely.
    """
    asyncio.run(_create_migration_db())
    try:
        _run_alembic("upgrade", "head")

        seeded = asyncio.run(_insert_sample_tech_items())
        seeded_ids = [item_id for item_id, _ in seeded]
        expected = dict(seeded)

        assert asyncio.run(_document_entities_table_exists()) is True, (
            "document_entities should exist right after upgrade head"
        )
        assert asyncio.run(_simhash_column_type()) == "bigint", (
            "tech_items.simhash should be a bigint column right after "
            "upgrade head"
        )

        _run_alembic("downgrade", "-1")

        assert asyncio.run(_document_entities_table_exists()) is False, (
            "downgrade -1 should drop document_entities"
        )
        assert asyncio.run(_simhash_column_type()) is None, (
            "downgrade -1 should drop tech_items.simhash"
        )

        survivors = asyncio.run(_fetch_tech_items(seeded_ids))
        assert survivors == expected, (
            "existing tech_items rows must keep their id and source_url "
            f"across the round trip; expected {expected}, got {survivors}"
        )

        _run_alembic("upgrade", "head")

        assert asyncio.run(_document_entities_table_exists()) is True, (
            "upgrade head should recreate document_entities"
        )
        assert asyncio.run(_simhash_column_type()) == "bigint", (
            "upgrade head should recreate tech_items.simhash"
        )

        survivors = asyncio.run(_fetch_tech_items(seeded_ids))
        assert survivors == expected
    finally:
        asyncio.run(_drop_migration_db())
