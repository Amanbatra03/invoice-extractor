import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.user import UserOut, RoleUpdateIn
from db.models import User
from db.session import get_db

router = APIRouter(prefix="/users", tags=["users"])
log = structlog.get_logger()

_VALID_ROLES = {"admin", "analyst", "viewer", "api_user"}


@router.get("", response_model=dict)
async def list_users(
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(User).where(User.tenant_id == uuid.UUID(user.tenant_id))
    )).scalars().all()
    return {"data": [UserOut.model_validate(r) for r in rows], "error": None, "request_id": None}


@router.patch("/{user_id}/role", response_model=dict)
async def update_role(
    user_id: uuid.UUID,
    body: RoleUpdateIn,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    if body.role not in _VALID_ROLES:
        raise HTTPException(400, f"Invalid role. Must be one of: {_VALID_ROLES}")
    target = await db.scalar(
        select(User).where(User.id == user_id, User.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not target:
        raise HTTPException(404, "User not found")
    target.role = body.role
    await db.commit()
    log.info("user.role_changed", target_user_id=str(user_id), new_role=body.role)
    return {"data": UserOut.model_validate(target), "error": None, "request_id": None}


@router.delete("/{user_id}", response_model=dict)
async def remove_user(
    user_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    target = await db.scalar(
        select(User).where(User.id == user_id, User.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not target:
        raise HTTPException(404, "User not found")
    if str(user_id) == user.id:
        raise HTTPException(400, "Cannot remove yourself")
    await db.delete(target)
    await db.commit()
    return {"data": {"deleted": True}, "error": None, "request_id": None}
