import uuid
from datetime import datetime
from pydantic import BaseModel


class JobOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    type: str
    status: str
    payload: dict | None
    result: dict | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class BatchJobResult(BaseModel):
    batch_job_id: uuid.UUID
    status: str
    total: int
    done: int
    failed: int
    results: list[dict]
