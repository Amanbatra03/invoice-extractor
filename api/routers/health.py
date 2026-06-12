import structlog
from fastapi import APIRouter
from redis import Redis

from api.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])
log = structlog.get_logger()


@router.get("")
async def health():
    settings = get_settings()
    db_ok = True
    redis_ok = True
    try:
        from db.session import get_engine
        import sqlalchemy
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        log.warning("health_db_failed", error=str(exc))
    try:
        r = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
        r.ping()
        r.close()
    except Exception as exc:
        redis_ok = False
        log.warning("health_redis_failed", error=str(exc))
    return {
        "data": {
            "status": "ok",
            "db": "ok" if db_ok else "degraded",
            "redis": "ok" if redis_ok else "degraded",
        },
        "error": None,
        "request_id": None,
    }


@router.get("/ready")
async def ready():
    return {"data": {"ready": True}, "error": None, "request_id": None}
