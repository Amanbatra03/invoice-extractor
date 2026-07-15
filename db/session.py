import functools

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@functools.lru_cache(maxsize=1)
def get_engine():
    from api.config import get_settings
    settings = get_settings()
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


@functools.lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        get_engine(), expire_on_commit=False, class_=AsyncSession
    )


async def get_db():
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
