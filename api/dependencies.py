import asyncio
import time as _time
from dataclasses import dataclass
from typing import Callable

import bcrypt
import structlog
import jwt as pyjwt
from fastapi import Depends, HTTPException, Request
from rq import Queue
from redis import Redis

from api.config import get_settings
from api.supabase_client import get_service_client, get_jwks_client

log = structlog.get_logger()


@dataclass
class CurrentUser:
    id: str
    tenant_id: str
    role: str
    email: str = ""


async def _verify_token(token: str, settings) -> dict:
    """Verify JWT: try ES256/RS256 via JWKS, fall back to HS256 with secret."""
    try:
        jwks = get_jwks_client()
        signing_key = await asyncio.to_thread(jwks.get_signing_key_from_jwt, token)
        return pyjwt.decode(
            token,
            signing_key,
            algorithms=["ES256", "RS256"],
            options={"verify_aud": False},
        )
    except Exception as exc:
        log.warning("jwt.jwks_verify_failed", error=str(exc))

    if settings.SUPABASE_JWT_SECRET:
        try:
            return pyjwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
        except Exception:
            pass

    raise HTTPException(401, "Cannot verify token")


async def _get_or_provision_user(sub: str, email: str) -> "CurrentUser":
    """Look up user in users table by sub; auto-provision tenant+user on first login."""
    def _lookup_by_sub():
        c = get_service_client()
        return c.table("users").select("id,tenant_id,email,role").eq("id", sub).limit(1).execute()

    res = await asyncio.to_thread(_lookup_by_sub)
    if res.data:
        row = res.data[0]
        return CurrentUser(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            role=str(row.get("role", "viewer")),
            email=str(row.get("email", email)),
        )

    def _lookup_by_email():
        c = get_service_client()
        return c.table("users").select("id,tenant_id,email,role").eq("email", email).limit(1).execute()

    res2 = await asyncio.to_thread(_lookup_by_email)
    if res2.data:
        row = res2.data[0]
        return CurrentUser(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            role=str(row.get("role", "viewer")),
            email=str(row.get("email", email)),
        )

    # Auto-provision: create tenant then user
    log.info("user.provisioning", email=email, sub=sub)
    tenant_name = email.split("@")[-1] if "@" in email else "default"

    def _provision():
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        c = get_service_client()
        t_res = c.table("tenants").insert({
            "name": tenant_name,
            "plan": "free",
            "created_at": now,
        }).execute()
        tenant = t_res.data[0]
        u_res = c.table("users").insert({
            "id": sub,
            "tenant_id": tenant["id"],
            "email": email,
            "role": "admin",
            "created_at": now,
        }).execute()
        return u_res.data[0], tenant["id"]

    try:
        user_row, tenant_id = await asyncio.to_thread(_provision)
        log.info("user.provisioned", email=email, tenant_id=str(tenant_id))
        return CurrentUser(
            id=str(user_row["id"]),
            tenant_id=str(user_row["tenant_id"]),
            role=str(user_row.get("role", "admin")),
            email=str(user_row.get("email", email)),
        )
    except Exception as exc:
        log.error("user.provision_failed", email=email, error=str(exc))
        raise HTTPException(500, "Failed to provision user account")


async def get_current_user(request: Request) -> CurrentUser:
    settings = get_settings()

    # ── Bearer JWT ────────────────────────────────────────────────
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        try:
            payload = await _verify_token(token, settings)
            sub = payload.get("sub", "")
            email = payload.get("email", "")

            if not sub:
                raise HTTPException(401, "Invalid token: missing sub")
            exp = payload.get("exp", 0)
            if exp and exp < _time.time():
                raise HTTPException(401, "Token expired")

            return await _get_or_provision_user(sub, email)

        except HTTPException:
            raise
        except Exception as exc:
            log.error("auth.unexpected_error", error=str(exc))
            raise HTTPException(401, f"Authentication failed: {exc}")

    # ── API Key ───────────────────────────────────────────────────
    api_key_raw = request.headers.get("X-API-Key", "")
    if api_key_raw:
        def _fetch_keys():
            c = get_service_client()
            return c.table("api_keys").select("*").eq("active", True).execute()

        key_res = await asyncio.to_thread(_fetch_keys)
        for key_row in (key_res.data or []):
            try:
                if bcrypt.checkpw(api_key_raw.encode(), key_row["key_hash"].encode()):
                    from datetime import datetime, timezone

                    def _touch_key(key_id):
                        c = get_service_client()
                        c.table("api_keys").update(
                            {"last_used_at": datetime.now(timezone.utc).isoformat()}
                        ).eq("id", key_id).execute()

                    await asyncio.to_thread(_touch_key, key_row["id"])
                    return CurrentUser(
                        id="api_key",
                        tenant_id=str(key_row["tenant_id"]),
                        role=key_row["role"],
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
