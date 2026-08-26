"""ARG-258 DB integration test — event layer migration round trip.

Verifies the parent AC "``alembic upgrade head`` → ``downgrade`` runs
without losing existing data" for revision ``855cb67b5096`` (add event layer
tables):

- ``upgrade head`` creates ``tech_events`` / ``event_documents`` / ``entities``
  / ``event_entities`` in one shot.
- downgrading back to ``855cb67b5096``'s parent revision drops all four
  (plus the ``entity_kind`` enum) and existing ``tech_items`` rows keep their
  id and ``source_url`` unchanged.
- ``upgrade head`` again brings the four tables back.

Downgrades to the absolute parent revision (``_EVENT_LAYER_DOWN_REVISION``)
rather than the relative ``downgrade -1`` this test originally used — ARG-262
stacked a new revision on top of ``855cb67b5096``, so "head" moved and a
relative ``-1`` from head would now only undo the ARG-262 revision, leaving
these four tables in place. See ``_EVENT_LAYER_DOWN_REVISION``'s comment.

Runs entirely against its own throwaway database, named
``argos_migration_test_<random>`` — **a fresh name per run**. It never touches
the dev DB (``argos``, which holds the real corpus) nor the pytest scratch DB
(``argos_test`` — that DB already has these four tables via
``Base.metadata.create_all`` at session start, so running ``alembic upgrade``
against it would fail with "already exists"; see ``tests/conftest.py``).

The name is randomized rather than fixed because this repo is worked on from
several git worktrees against one shared Docker Postgres. With a fixed name,
two test runs started from different worktrees would force-drop each other's
database mid-migration. Randomizing also means this module only ever drops a
database **it created itself**: setup does a bare ``CREATE DATABASE`` with no
preceding force-drop, so a name that somehow already exists fails loudly
instead of being destroyed silently.

A run killed between CREATE and teardown leaks one empty database. They are all
prefixed ``argos_migration_test_`` — ``psql -l`` shows them, and dropping any of
them is always safe.

Skips cleanly when Postgres is unreachable, matching release CI (no DB
service) — see ``tests/test_cli_backfill_images_db.py`` for the same pattern.
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

# Captured at import time, same pattern as tests/test_cli_backfill_images_db.py.
# conftest.py has already pinned POSTGRES_DB to the pytest scratch DB by the
# time this module is imported; only host/port/credentials from it are used
# below — the throwaway migration DB name is substituted explicitly.
_BASE_URL: str = settings.database_url
_MIGRATION_DB_PREFIX = "argos_migration_test"
# 실행마다 새 이름 — 같은 Postgres를 공유하는 다른 워크트리의 실행과 겹치지
# 않게. 덕분에 이 모듈은 자기가 만든 DB만 드롭한다 (모듈 docstring 참고).
_MIGRATION_DB_NAME = f"{_MIGRATION_DB_PREFIX}_{uuid.uuid4().hex[:12]}"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVENT_LAYER_TABLES = ["tech_events", "event_documents", "entities", "event_entities"]
# 리비전 855cb67b5096(이 테스트가 검증하는 이벤트 레이어 리비전)의 down_revision.
# 상대 경로 "-1" 대신 이 절대 리비전으로 내려간다 — ARG-262가 그 위에 새
# 리비전을 쌓은 뒤로 "head"가 더 이상 855cb67b5096이 아니라서, "-1"은 이제
# ARG-262 리비전 하나만 되돌리고 이벤트 레이어 테이블은 그대로 남긴다. 절대
# 리비전 타깃은 앞으로 head 위에 몇 개가 더 쌓이든 정확히 이 리비전만 되돌린다.
_EVENT_LAYER_DOWN_REVISION = "16778301d264"


def _assert_migration_db_is_disposable(
    migration_db: str, dev_db: str, scratch_db: str
) -> None:
    """``migration_db``가 지워도 되는 이름인지 확인한다 — 아니면 던진다.

    teardown은 이 이름에 대고 ``DROP DATABASE ... WITH (FORCE)``를 실행한다.
    이름이 개발자의 진짜 dev DB나 pytest 스크래치 DB와 겹치면 그 DB가 통째로
    날아간다.

    1차 방어는 이름 자체다 — ``_MIGRATION_DB_NAME``은 실행마다 랜덤 접미사를
    달아서 어떤 실제 DB와도 겹치지 않는다. 이 함수는 그 뒤를 받치는 마지막
    방어선이다: 나중에 누군가 이름을 고정값으로 되돌리면 그 순간 여기서 걸린다.

    이 검사가 따로 필요한 이유는 conftest의 기존 충돌 검사가 이 이름을 보지
    않기 때문이다. 그 검사는 ``ARGOS_TEST_DB_NAME``(스크래치)만 dev DB 이름과
    비교한다. 즉 이름을 ``argos_migration_test``로 고정해 두면, ``POSTGRES_DB``
    가 하필 그 값인 개발자는 검사를 그대로 통과한 뒤(스크래치 ``argos_test``와는
    다르니까) 자기 DB가 드롭된다.

    skip이 아니라 예외로 멈춘다 — 조용히 건너뛰면 잘못된 설정이 안 보이고,
    이건 파괴적 작업 직전의 마지막 방어선이다.
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
            "pgvector DB not reachable — skipping ARG-258 event-layer "
            "migration round-trip test (start the Docker DB to run it)"
        )
    # DB가 살아 있을 때만 — 여기서부터 CREATE/DROP이 실제로 나간다.
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
    """Insert two rows standing in for the dev DB's 556 real documents."""
    import asyncpg

    params = _connect_params()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    rows: list[tuple[uuid.UUID, str]] = []
    try:
        for i in range(2):
            item_id = uuid.uuid4()
            source_url = f"https://arg258-migration-test.example/{item_id}"
            await conn.execute(
                """
                INSERT INTO tech_items
                    (id, title, source_url, raw_content, category, trust_score,
                     created_at, updated_at)
                VALUES ($1, $2, $3, $4, 'Mainstream', 0.5, now(), now())
                """,
                item_id,
                f"ARG-258 migration fixture {i}",
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


async def _existing_tables(names: list[str]) -> set[str]:
    import asyncpg

    params = _connect_params()
    conn = await asyncpg.connect(database=_MIGRATION_DB_NAME, **params)
    try:
        records = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "AND tablename = ANY($1::text[])",
            names,
        )
    finally:
        await conn.close()
    return {r["tablename"] for r in records}


def _run_alembic(fn_name: str, revision: str) -> None:
    """Run one alembic command against the throwaway migration DB only.

    ``alembic/env.py`` reads ``settings.database_url`` fresh on every
    invocation, so this temporarily points the shared ``settings`` singleton's
    ``POSTGRES_DB`` at ``argos_migration_test`` for the duration of the call,
    then restores the original value in a ``finally`` — no other test module
    (nor a crash mid-test) can leave the process pointed at the throwaway DB.

    Deliberately constructed *without* loading ``alembic.ini`` (no ``file_``
    passed to ``Config``): when ``Config.config_file_name`` is set,
    ``alembic/env.py`` calls ``logging.config.fileConfig()``, which defaults
    to ``disable_existing_loggers=True`` and silently tears down every logger
    already configured in this pytest process — observed in practice to break
    an unrelated logging assertion in ``tests/test_progress.py`` when both ran
    in the same session. ``script_location`` is the only ini option
    ``env.py``/``ScriptDirectory`` need, and it is set explicitly below.
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


def test_event_layer_migration_round_trip():
    """upgrade head -> downgrade -1 -> upgrade head loses no data (ARG-258).

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

        before_downgrade = asyncio.run(_existing_tables(_EVENT_LAYER_TABLES))
        assert before_downgrade == set(_EVENT_LAYER_TABLES), (
            "all four event-layer tables should exist right after "
            f"upgrade head, found: {before_downgrade}"
        )

        _run_alembic("downgrade", _EVENT_LAYER_DOWN_REVISION)

        remaining = asyncio.run(_existing_tables(_EVENT_LAYER_TABLES))
        assert remaining == set(), (
            "downgrading to the event-layer revision's parent should drop "
            f"all four event-layer tables, but these survived: {remaining}"
        )

        survivors = asyncio.run(_fetch_tech_items(seeded_ids))
        assert survivors == expected, (
            "existing tech_items rows must keep their id and source_url "
            f"across the round trip; expected {expected}, got {survivors}"
        )

        _run_alembic("upgrade", "head")

        after_reupgrade = asyncio.run(_existing_tables(_EVENT_LAYER_TABLES))
        assert after_reupgrade == set(_EVENT_LAYER_TABLES), (
            "upgrade head should recreate all four event-layer tables, "
            f"found: {after_reupgrade}"
        )
    finally:
        asyncio.run(_drop_migration_db())
