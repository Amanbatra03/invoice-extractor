import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import require_roles, CurrentUser
from api.schemas.job import JobOut
from api.supabase_client import get_service_client

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=dict)
async def get_job(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
):
    def _get():
        c = get_service_client()
        return c.table("jobs").select("*").eq("id", str(job_id)).eq("tenant_id", user.tenant_id).limit(1).execute()

    res = await asyncio.to_thread(_get)
    if not res.data:
        raise HTTPException(404, "Job not found")
    return {"data": JobOut.model_validate(res.data[0]), "error": None, "request_id": None}


@router.get("", response_model=dict)
async def list_jobs(
    status: str | None = Query(None),
    type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: CurrentUser = Depends(require_roles("admin", "analyst")),
):
    def _list():
        c = get_service_client()
        q = c.table("jobs").select("*").eq("tenant_id", user.tenant_id).order("created_at", desc=True)
        if status:
            q = q.eq("status", status)
        if type:
            q = q.eq("type", type)
        return q.limit(limit).execute()

    res = await asyncio.to_thread(_list)
    rows = res.data or []
    return {
        "data": {"items": [JobOut.model_validate(r) for r in rows]},
        "error": None,
        "request_id": None,
    }
