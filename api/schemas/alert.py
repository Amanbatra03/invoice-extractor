import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    severity: str
    source: str
    event: str
    detail: str
    context: dict | None
    fingerprint: str
    delivery_status: str
    delivery_attempts: int
    last_error: str | None
    delivered_at: datetime | None
    created_at: datetime
