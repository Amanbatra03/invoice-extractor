import time as _time
import uuid
from dataclasses import dataclass
from typing import Callable

import bcrypt
import structlog
import jwt as pyjwt
from fastapi import Depends, HTTPException, Request
from rq import Queue
from redis import Redis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

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
    except Exception as exc:
        raise HTTPException(401, f"Invalid token: {exc}")


async def _verify_via_db(db: AsyncSession, token: str) -> tuple[str, str]:
    """
    Fallback auth when SUPABASE_JWT_SECRET is absent/wrong.
    Decodes JWT structure without signature verification, then confirms
    the sub exists as an email-confirmed user in auth.users via the
    existing DB session (no outbound HTTP required).
    Token expiry is still enforced.
    """
    try:
        payload = pyjwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["HS256", "RS256"],
        )
    except pyjwt.DecodeError:
        raise HTTPException(401, "Malformed token")

    sub = payload.get("sub", "")
    email = payload.get("email", "")
    exp = payload.get("exp", 0)

    if exp and exp < _time.time():
        raise HTTPException(401, "Token expired")
    if not sub:
        raise HTTPException(401, "Invalid token: missing sub")

    row = (await db.execute(
        text(
            "SELECT id::text, email FROM auth.users "
            "WHERE id::text = :sub AND email_confirmed_at IS NOT NULL LIMIT 1"
        ),
        {"sub": sub},
    )).fetchone()

    if not row:
        raise HTTPException(401, "Invalid token: unrecognized user")

    return str(row.id), str(row.email or email)


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


async def _authenticate_bearer(token: str, db: AsyncSession, settings) -> CurrentUser:
    sub = ""
    email = ""

    # Step 1: local JWT verification with secret (fast, no network required)
    if settings.SUPABASE_JWT_SECRET:
        try:
            payload = verify_supabase_jwt(token, settings.SUPABASE_JWT_SECRET)
            sub = payload.get("sub", "")
            email = payload.get("email", "")
            app_meta = payload.get("app_metadata", {})
            tenant_id = app_meta.get("tenant_id", "")
            role = app_meta.get("role", "viewer")
            if tenant_id:
                return CurrentUser(id=sub, tenant_id=tenant_id, role=role, email=email)
            # tenant_id absent in token → fall through to DB lookup
        except HTTPException:
            sub = ""
            email = ""
            log.warning("jwt.local_verify_failed_using_db_fallback")

    # Step 2: DB-based fallback — decode JWT structure + confirm user in auth.users
    if not sub:
        sub, email = await _verify_via_db(db, token)

    # Look up user in our users table; auto-provision admin on first login
    user_row = await db.scalar(select(User).where(User.email == email))
    if not user_row:
        user_row = await _provision_user(db, sub, email)

    return CurrentUser(
        id=str(user_row.id),
        tenant_id=str(user_row.tenant_id),
        role=user_row.role,
        email=user_row.email,
    )


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    settings = get_settings()

    # ── Bearer JWT ────────────────────────────────────────────────
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        try:
            return await _authenticate_bearer(token, db, settings)
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(503, f"Service temporarily unavailable: {exc}")
        except Exception as exc:
            log.error("auth.unexpected_error", error=str(exc))
            raise HTTPException(401, f"Authentication failed: {exc}")

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
