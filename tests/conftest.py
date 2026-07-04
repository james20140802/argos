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

3. 위 (1)에서 ``POSTGRES_DB``를 덮어쓰기 **직전**, 실제 dev DB명(env var →
   .env 파일 → 하드코드 기본값 ``argos`` 순)을 별도로 읽어 스크래치 DB명과
   비교한다. 둘이 같으면(예: 개발자 셸에 ``ARGOS_TEST_DB_NAME=argos``가 export
   돼 있는 경우) pytest 컬렉션 단계에서 즉시 ``RuntimeError``로 중단한다 —
   그렇지 않으면 (2)의 세션 fixture가 dev DB를 DROP해 버린다.
"""

from __future__ import annotations

import functools
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# --------------------------------------------------------------------- #
# Isolated scratch DB name — must be set before `argos.config` is ever
# imported (by us or by any test module) since `Settings()`/`Secrets()` are
# module-level singletons resolved once at import time.
# --------------------------------------------------------------------- #
TEST_DB_NAME = os.environ.get("ARGOS_TEST_DB_NAME", "argos_test")

# Well-known dev DB name (argos.config.Secrets.POSTGRES_DB default). Hard
# reject this regardless of what the real .env resolves to — it's the one
# name we know for certain is never safe to DROP/CREATE as a scratch DB.
_HARDCODED_DEV_DB_NAME = "argos"


def _resolve_env_file() -> Path | None:
    """Mirror ``argos.config._resolve_env_file`` without importing
    ``argos.config`` — importing it here would freeze its module-level
    ``settings`` singleton against whatever ``POSTGRES_DB`` is in
    ``os.environ`` *before* we've had a chance to override it below, which
    would silently defeat this file's entire isolation mechanism for every
    test module that does ``from argos.config import settings``.
    """
    env_file_override = os.environ.get("ARGOS_ENV_FILE")
    if env_file_override:
        return Path(env_file_override)

    xdg = os.environ.get("XDG_CONFIG_HOME")
    xdg_base = Path(xdg) if xdg else Path.home() / ".config"
    xdg_path = xdg_base / "argos" / ".env"
    if xdg_path.exists():
        return xdg_path

    cwd_env = Path(".env")
    if cwd_env.exists():
        return cwd_env

    return None


def _real_dev_db_name() -> str:
    """Resolve the dev DB name the app would actually connect to, following
    the same precedence as ``argos.config.Secrets`` (explicit env var > .env
    file > hardcoded field default) — without importing ``argos.config``.
    """
    env_value = os.environ.get("POSTGRES_DB")
    if env_value:
        return env_value

    env_file = _resolve_env_file()
    if env_file is not None:
        from dotenv import dotenv_values

        file_value = dotenv_values(env_file).get("POSTGRES_DB")
        if file_value:
            return file_value

    return _HARDCODED_DEV_DB_NAME  # argos.config.Secrets.POSTGRES_DB default


_real_dev_db_name_resolved = _real_dev_db_name()
if TEST_DB_NAME in (_HARDCODED_DEV_DB_NAME, _real_dev_db_name_resolved):
    raise RuntimeError(
        f"ARGOS_TEST_DB_NAME={TEST_DB_NAME!r} collides with the dev database "
        f"name ({_real_dev_db_name_resolved!r}). The isolated-test-DB fixture "
        f"in this file runs `DROP DATABASE IF EXISTS \"{TEST_DB_NAME}\" WITH "
        f"(FORCE)` at session start — running it against the dev DB would "
        f"destroy it. Set ARGOS_TEST_DB_NAME to a distinct scratch DB name "
        f"(default: argos_test)."
    )

os.environ["POSTGRES_DB"] = TEST_DB_NAME


def _server_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if a TCP connection to host:port succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@functools.lru_cache(maxsize=None)
def _admin_reset_capable(
    host: str,
    port: int,
    user: str | None,
    password: str | None,
    timeout: float = 2.0,
) -> bool:
    """True only if the configured credentials can actually open an
    authenticated admin connection to the ``postgres`` maintenance DB *and*
    are allowed to create databases (``rolcreatedb`` or superuser).

    TCP reachability alone is not enough: an unrelated local Postgres may be
    listening on the same host/port while the Argos credentials fail to
    authenticate or lack ``CREATEDB`` (common on developer machines). Probing
    the full precondition here means the ``_isolated_test_database`` reset and
    every DB-backed test's ``skipif(not db_reachable(...))`` guard agree — so
    such an environment cleanly *skips* DB tests instead of crashing the whole
    suite (including pure unit tests) at session setup.

    Memoized: credentials are constant for a run, so the network probe runs at
    most once even though ``db_reachable`` is evaluated per DB-backed module.
    """
    import asyncio

    async def _probe() -> bool:
        import asyncpg

        try:
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database="postgres",
                timeout=timeout,
            )
        except (OSError, asyncpg.PostgresError, asyncio.TimeoutError):
            return False
        try:
            # rolsuper implies CREATEDB even when rolcreatedb is false.
            return bool(
                await conn.fetchval(
                    "SELECT rolcreatedb OR rolsuper FROM pg_roles "
                    "WHERE rolname = current_user"
                )
            )
        except asyncpg.PostgresError:
            return False
        finally:
            await conn.close()

    try:
        return asyncio.run(_probe())
    except Exception:
        return False


def db_reachable(url: str) -> bool:
    """True if the Postgres server behind ``url`` (a SQLAlchemy URL string) is
    genuinely usable for the isolated-test-DB workflow: TCP-reachable **and**
    the configured credentials can authenticate and create databases.

    This is stricter than a bare TCP probe on purpose — see
    ``_admin_reset_capable``. DB-backed test modules import this as their
    ``skipif`` guard, so gating on the same precondition the session fixture
    needs keeps them in lockstep: a foreign/unauthorized Postgres on the
    configured host/port makes DB tests skip rather than error.
    """
    from sqlalchemy.engine.url import make_url

    parsed = make_url(url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432
    if not _server_reachable(host, port):
        return False
    return _admin_reset_capable(host, port, parsed.username, parsed.password)


@pytest.fixture(scope="session", autouse=True)
def _isolated_test_database():
    """(Re)create the scratch test DB once per pytest session.

    No-op unless the configured Postgres is genuinely usable — i.e. the same
    ``db_reachable`` precondition (TCP + auth + CREATEDB) that DB-backed tests
    gate their ``skipif`` on. This means:

    * Release CI (no DB service): TCP fails → no-op, DB tests self-skip.
    * A developer machine with an *unrelated* Postgres on the configured
      host/port (mismatched Argos creds, or an app user without ``CREATEDB``):
      the auth/privilege probe fails → no-op, DB tests self-skip. The suite is
      never taken down at session setup — pure unit tests still run.

    The DROP/CREATE is additionally wrapped so any residual admin error
    (e.g. a race that revokes ``CREATEDB`` between probe and reset) degrades to
    a warning + no-op rather than crashing every test's setup.
    """
    from argos.config import settings

    db_url = settings.database_url  # already resolves to the scratch DB
    from sqlalchemy.engine.url import make_url

    parsed = make_url(db_url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432

    if not db_reachable(db_url):
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

    try:
        asyncio.run(_reset())
    except Exception as exc:  # pragma: no cover - defensive setup guard
        import asyncpg

        if isinstance(exc, (OSError, asyncpg.PostgresError)):
            import warnings

            warnings.warn(
                f"Skipping isolated-test-DB reset: admin reset failed against "
                f"{host}:{port} ({type(exc).__name__}: {exc}). DB-backed tests "
                f"will fail/skip via their own guards; unit tests are "
                f"unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )
            yield
            return
        raise
    yield


@pytest.fixture
def sample_uuid():
    """테스트용 고정 UUID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_datetime():
    """테스트용 고정 datetime."""
    return datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
