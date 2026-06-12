import uuid
from dataclasses import dataclass
from typing import Callable

import bcrypt
import structlog
from fastapi import Depends, HTTPException, Request
from jose import JWTError, jwt
from rq import Queue
from redis import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from db.models import ApiKey, User
from db.session import get_db

log = structlog.get_logger()


@dataclass
class CurrentUser:
    id: str
    tenant_id: str
    role: str
    email: str = ""


def verify_supabase_jwt(token: str, secret: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        return payload
    except JWTError as exc:
        raise HTTPException(401, f"Invalid token: {exc}")


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    settings = get_settings()

    # --- Bearer JWT path ---
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        try:
            payload = verify_supabase_jwt(token, settings.SUPABASE_JWT_SECRET)
        except HTTPException:
            raise

        sub = payload.get("sub", "")
        app_meta = payload.get("app_metadata", {})
        tenant_id = app_meta.get("tenant_id", "")
        role = app_meta.get("role", "viewer")
        email = payload.get("email", "")

        if not tenant_id:
            raise HTTPException(401, "Token missing tenant_id in app_metadata")

        return CurrentUser(id=sub, tenant_id=tenant_id, role=role, email=email)

    # --- API Key path ---
    api_key_raw = request.headers.get("X-API-Key", "")
    if api_key_raw:
        rows = (await db.execute(
            select(ApiKey).where(ApiKey.active == True)
        )).scalars().all()

        for key_row in rows:
            try:
                if bcrypt.checkpw(api_key_raw.encode(), key_row.key_hash.encode()):
                    from datetime import datetime, timezone
                    key_row.last_used_at = datetime.now(timezone.utc)
                    await db.commit()
                    return CurrentUser(
                        id="api_key",
                        tenant_id=str(key_row.tenant_id),
                        role=key_row.role,
                        email="",
                    )
            except Exception:
                continue

        raise HTTPException(401, "Invalid API key")

    raise HTTPException(401, "Authentication required (Bearer token or X-API-Key header)")


def require_roles(*roles: str) -> Callable:
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(
                403,
                f"Role '{user.role}' not permitted. Required: {list(roles)}",
            )
        return user
    return _check


def get_queue() -> Queue:
    settings = get_settings()
    conn = Redis.from_url(settings.REDIS_URL)
    return Queue("invoice-jobs", connection=conn)
