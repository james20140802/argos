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
