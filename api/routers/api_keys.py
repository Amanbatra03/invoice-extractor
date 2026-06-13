import secrets
import uuid
from datetime import datetime, timezone

import bcrypt
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from db.models import ApiKey
from db.session import get_db

router = APIRouter(prefix="/api-keys", tags=["api-keys"])
log = structlog.get_logger()


class ApiKeyIn(BaseModel):
    name: str


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    role: str
    active: bool
    last_used_at: datetime | None
    created_at: datetime


@router.post("", response_model=dict)
async def create_api_key(
    body: ApiKeyIn,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    import asyncio

    raw_key = secrets.token_urlsafe(32)
    key_hash = await asyncio.to_thread(bcrypt.hashpw, raw_key.encode(), bcrypt.gensalt())
    key = ApiKey(
        tenant_id=uuid.UUID(user.tenant_id),
        name=body.name,
        key_hash=key_hash.decode(),
        role="api_user",
        active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(key)
    await db.commit()
    await db.refresh(key)
    log.info("api_key.created", key_id=str(key.id), name=body.name)
    return {
        "data": {**ApiKeyOut.model_validate(key).model_dump(), "raw_key": raw_key},
        "error": None,
        "request_id": None,
    }


@router.get("", response_model=dict)
async def list_api_keys(
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(ApiKey).where(ApiKey.tenant_id == uuid.UUID(user.tenant_id))
    )).scalars().all()
    return {"data": [ApiKeyOut.model_validate(r) for r in rows], "error": None, "request_id": None}


@router.delete("/{key_id}", response_model=dict)
async def revoke_api_key(
    key_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    key = await db.scalar(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not key:
        raise HTTPException(404, "API key not found")
    key.active = False
    await db.commit()
    return {"data": {"revoked": True}, "error": None, "request_id": None}
