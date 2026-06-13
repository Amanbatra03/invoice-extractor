import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from db.models import AuditLog
from db.session import get_db

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=dict)
async def list_audit_log(
    limit: int = Query(100, ge=1, le=500),
    action: str | None = Query(None),
    user_id: uuid.UUID | None = Query(None),
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(AuditLog)
        .where(AuditLog.tenant_id == uuid.UUID(user.tenant_id))
        .order_by(AuditLog.created_at.desc())
    )
    if action:
        q = q.where(AuditLog.action == action)
    if user_id:
        q = q.where(AuditLog.user_id == user_id)
    rows = (await db.execute(q.limit(limit))).scalars().all()
    return {
        "data": [
            {
                "id": str(r.id),
                "action": r.action,
                "user_id": str(r.user_id) if r.user_id else None,
                "resource_type": r.resource_type,
                "resource_id": str(r.resource_id) if r.resource_id else None,
                "metadata": r.meta,
                "created_at": r.created_at,
            }
            for r in rows
        ],
        "error": None,
        "request_id": None,
    }
