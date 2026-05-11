from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from argos.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_session():
    """비동기 DB 세션을 제공하는 제너레이터."""
    async with AsyncSessionLocal() as session:
        yield session


def rebuild(env_path: Path | None = None) -> None:
    """Rebuild the module-level engine and session factory from current env.

    Call this after any operation that writes new ``POSTGRES_*`` values to
    ``.env`` (e.g. the init wizard's infra step) so subsequent callers — in
    particular the healthcheck's ``db_ping`` — see the fresh credentials
    rather than the stale module-load snapshot.

    If ``env_path`` is given its contents are loaded into ``os.environ``
    (with override) before the settings singleton is refreshed, ensuring the
    new values are visible to pydantic-settings' ``Secrets`` constructor.
    Safe to call multiple times.
    """
    global engine, AsyncSessionLocal  # noqa: PLW0603

    if env_path is not None:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=True)

    # Re-instantiate Secrets so pydantic-settings re-reads the environment.
    from argos.config import Secrets

    settings.secrets = Secrets()

    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
