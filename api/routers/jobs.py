import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.job import JobOut
from db.models import Job
from db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])
log = structlog.get_logger()


@router.get("/{job_id}", response_model=dict)
async def get_job(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    job = await db.scalar(
        select(Job).where(
            Job.id == job_id,
            Job.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not job:
        raise HTTPException(404, "Job not found")
    return {"data": JobOut.model_validate(job), "error": None, "request_id": None}


@router.get("", response_model=dict)
async def list_jobs(
    status: str | None = Query(None),
    type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: CurrentUser = Depends(require_roles("admin", "analyst")),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(Job)
        .where(Job.tenant_id == uuid.UUID(user.tenant_id))
        .order_by(Job.created_at.desc())
    )
    if status:
        q = q.where(Job.status == status)
    if type:
        q = q.where(Job.type == type)
    rows = (await db.execute(q.limit(limit))).scalars().all()
    return {
        "data": [JobOut.model_validate(r) for r in rows],
        "error": None,
        "request_id": None,
    }
