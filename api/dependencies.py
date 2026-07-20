import asyncio
import uuid
from dataclasses import dataclass
from typing import Callable

import bcrypt
import structlog
import jwt as pyjwt
from fastapi import Depends, HTTPException, Request
from rq import Queue
from redis import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_client

from api.config import get_settings
from db.models import ApiKey, Tenant, User
from db.session import get_db

log = structlog.get_logger()


@dataclass
class CurrentUser:
    id: str
    tenant_id: str
    role: str
    email: str = ""


def verify_supabase_jwt(token: str, secret: str) -> dict:
    """Verify a Supabase JWT locally using the project secret."""
    try:
        return pyjwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except pyjwt.PyJWTError as exc:
        raise HTTPException(401, f"Invalid token: {exc}")


async def _verify_via_supabase_api(url: str, anon_key: str, token: str):
    """Verify token via Supabase Auth API — no JWT secret required."""
    sb = create_client(url, anon_key)
    response = await asyncio.to_thread(sb.auth.get_user, token)
    return response.user


async def _provision_user(db: AsyncSession, sub: str, email: str) -> User:
    """Auto-create a tenant + admin user on first login."""
    tenant_name = email.split("@")[-1] if "@" in email else "default"
    tenant = Tenant(name=tenant_name, plan="free")
    db.add(tenant)
    await db.flush()

    user_row = User(
        id=uuid.UUID(sub) if sub else uuid.uuid4(),
        tenant_id=tenant.id,
        email=email,
        role="admin",
    )
    db.add(user_row)
    await db.commit()
    await db.refresh(user_row)
    log.info("user.provisioned", email=email, tenant_id=str(tenant.id))
    return user_row


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    settings = get_settings()

    # ── Bearer JWT ────────────────────────────────────────────────
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        sub = ""
        email = ""

        if settings.SUPABASE_JWT_SECRET:
            # Local verification (fast; also the test path — tests patch this function)
            try:
                payload = verify_supabase_jwt(token, settings.SUPABASE_JWT_SECRET)
                sub = payload.get("sub", "")
                email = payload.get("email", "")
                app_meta = payload.get("app_metadata", {})
                tenant_id = app_meta.get("tenant_id", "")
                role = app_meta.get("role", "viewer")

                if tenant_id:
                    # JWT already carries tenant claims (test mocks / custom-provisioned)
                    return CurrentUser(id=sub, tenant_id=tenant_id, role=role, email=email)
                # else fall through to DB lookup below
            except HTTPException:
                # Local verification failed (wrong secret / alg mismatch) — fall back to API
                log.warning("jwt.local_verify_failed_fallback_to_api")
                try:
                    sb_user = await _verify_via_supabase_api(
                        settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY, token
                    )
                    if not sb_user:
                        raise HTTPException(401, "Invalid token")
                    sub = str(sb_user.id)
                    email = sb_user.email or ""
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(401, f"Invalid token: {exc}")
        else:
            # No secret configured → verify via Supabase Auth API
            try:
                sb_user = await _verify_via_supabase_api(
                    settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY, token
                )
                if not sb_user:
                    raise HTTPException(401, "Invalid token")
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(401, f"Invalid token: {exc}")
            sub = str(sb_user.id)
            email = sb_user.email or ""

        # Look up user in DB; auto-provision on first login
        user_row = await db.scalar(select(User).where(User.email == email))
        if not user_row:
            user_row = await _provision_user(db, sub, email)

        return CurrentUser(
            id=str(user_row.id),
            tenant_id=str(user_row.tenant_id),
            role=user_row.role,
            email=user_row.email,
        )

    # ── API Key ───────────────────────────────────────────────────
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
