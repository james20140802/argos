"""테스트 공통 fixture 정의.

ARG-191: DB 테스트 격리
-----------------------
DB-backed 테스트는 개발자의 실제 dev DB(``settings.database_url`` → 기본
``argos`` DB)가 아니라 같은 Postgres 서버 위의 별도 스크래치 DB(기본
``argos_test``)만 사용해야 한다. 이를 강제하기 위해:

1. 이 파일의 최상단(다른 어떤 ``argos.*`` 모듈보다 먼저 실행됨)에서
   ``POSTGRES_DB`` 환경변수를 스크래치 DB 이름으로 덮어쓴다. pytest는 실제
   테스트 모듈을 임포트하기 전에 conftest.py들을 먼저 로드하므로, 이후
   ``argos.config.settings``(그리고 각 테스트 파일이 import 시점에 캡처하는
   ``settings.database_url``)는 항상 스크래치 DB를 가리키게 된다.
2. 세션 스코프 autouse fixture가 (Postgres 서버가 열려 있을 때만) 스크래치
   DB를 DROP/CREATE로 매 세션 초기화하고, pgvector/uuid-ossp 익스텐션을 켠
   뒤 ORM 모델 메타데이터로 스키마를 만든다. 서버 자체가 열려 있지 않으면
   (릴리즈 CI처럼 Postgres가 없는 환경) 아무 것도 하지 않고 조용히 넘어가서
   각 DB 테스트 파일의 기존 ``skipif`` self-skip 메커니즘이 정상 동작한다.

dev DB에는 이 conftest에서 절대 쓰기(CREATE/DROP/INSERT/...)를 하지 않는다 —
건드리는 것은 오직 ``POSTGRES_DB``로 지정된 스크래치 DB 뿐이다.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import datetime, timezone

import pytest

# --------------------------------------------------------------------- #
# Isolated scratch DB name — must be set before `argos.config` is ever
# imported (by us or by any test module) since `Settings()`/`Secrets()` are
# module-level singletons resolved once at import time.
# --------------------------------------------------------------------- #
TEST_DB_NAME = os.environ.get("ARGOS_TEST_DB_NAME", "argos_test")
os.environ["POSTGRES_DB"] = TEST_DB_NAME


def _server_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if a TCP connection to host:port succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def db_reachable(url: str) -> bool:
    """True if the Postgres server behind ``url`` (a SQLAlchemy URL string)
    accepts TCP connections.

    Only checks TCP reachability — not auth or database existence — matching
    the pre-existing per-file skip helpers this centralizes. Shared here so
    individual DB-backed test modules don't each redefine the same helper.
    """
    from sqlalchemy.engine.url import make_url

    parsed = make_url(url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432
    return _server_reachable(host, port)


@pytest.fixture(scope="session", autouse=True)
def _isolated_test_database():
    """(Re)create the scratch test DB once per pytest session.

    No-op when Postgres is unreachable (e.g. Release CI, which runs pytest
    with no DB service) — DB-backed tests self-skip via their own
    ``skipif(not db_reachable(...))`` guards, unaffected by this fixture.
    """
    from argos.config import settings

    db_url = settings.database_url  # already resolves to the scratch DB
    from sqlalchemy.engine.url import make_url

    parsed = make_url(db_url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432

    if not _server_reachable(host, port):
        yield
        return

    import asyncio

    async def _reset() -> None:
        import asyncpg

        admin_conn = await asyncpg.connect(
            host=host,
            port=port,
            user=parsed.username,
            password=parsed.password,
            database="postgres",
        )
        try:
            # FORCE disconnects any lingering sessions from a prior crashed
            # run so DROP DATABASE never hangs/errors on "in use" (PG13+).
            await admin_conn.execute(
                f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)'
            )
            await admin_conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
        finally:
            await admin_conn.close()

        test_conn = await asyncpg.connect(
            host=host,
            port=port,
            user=parsed.username,
            password=parsed.password,
            database=TEST_DB_NAME,
        )
        try:
            await test_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await test_conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        finally:
            await test_conn.close()

        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        from argos.models.base import Base

        engine = create_async_engine(db_url, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(_reset())
    yield


@pytest.fixture
def sample_uuid():
    """테스트용 고정 UUID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_datetime():
    """테스트용 고정 datetime."""
    return datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
