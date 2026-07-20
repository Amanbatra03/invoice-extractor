import re
import ssl as _ssl
import functools
import structlog

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

log = structlog.get_logger()

# db.[ref].supabase.co is IPv6-only on Render free tier.
# The session pooler (aws-0-[region].pooler.supabase.com) has IPv4 records.
_DIRECT_PATTERN = re.compile(
    r"(postgresql(?:\+asyncpg)?://)(postgres):([^@]*)@db\.([a-z0-9]+)\.supabase\.co(?::\d+)?(/[^?]*)?"
)
_POOLER_REGION = "ap-northeast-1"


def _to_session_pooler(url: str) -> tuple[str, dict]:
    """
    Rewrite a Supabase direct-connection URL to the session-pooler URL.
    Returns (url, connect_args).
    """
    m = _DIRECT_PATTERN.match(url)
    if not m:
        return url, {}
    scheme, _, password, ref, db = m.groups()
    db = db or "/postgres"
    pooler = f"aws-0-{_POOLER_REGION}.pooler.supabase.com"
    new_url = f"{scheme}postgres.{ref}:{password}@{pooler}:5432{db}"
    log.info("db.using_session_pooler", pooler=pooler, ref=ref)
    return new_url, {"ssl": _ssl.create_default_context()}


@functools.lru_cache(maxsize=1)
def get_engine():
    from api.config import get_settings
    settings = get_settings()
    url, connect_args = _to_session_pooler(settings.DATABASE_URL)
    return create_async_engine(
        url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        connect_args=connect_args,
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
            try:
                await session.rollback()
            except Exception:
                pass
            raise
