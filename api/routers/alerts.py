from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.alert import AlertOut
from db.models import Alert
from db.session import get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=dict)
async def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    severity: str | None = Query(None),
    source: str | None = Query(None),
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    q = select(Alert).order_by(Alert.created_at.desc())
    if severity:
        q = q.where(Alert.severity == severity)
    if source:
        q = q.where(Alert.source == source)
    rows = (await db.execute(q.limit(limit).offset(offset))).scalars().all()
    return {
        "data": [AlertOut.model_validate(r).model_dump(mode="json") for r in rows],
        "error": None,
        "request_id": None,
    }
