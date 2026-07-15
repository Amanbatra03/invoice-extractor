import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, field_validator

_VALID_EVENTS = {
    "extraction.completed", "batch.done",
    "discrepancy.detected", "ingest.failed",
}


class WebhookIn(BaseModel):
    url: str
    events: list[str]
    secret: str

    @field_validator("events")
    @classmethod
    def events_not_empty(cls, v):
        if not v:
            raise ValueError("events must not be empty")
        invalid = set(v) - _VALID_EVENTS
        if invalid:
            raise ValueError(f"Unknown events: {invalid}. Valid: {_VALID_EVENTS}")
        return v


class WebhookOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    url: str
    events: list[str]
    active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WebhookPatch(BaseModel):
    url: str | None = None
    events: list[str] | None = None
    active: bool | None = None

    @field_validator("events")
    @classmethod
    def events_valid(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("events must not be empty")
        invalid = set(v) - _VALID_EVENTS
        if invalid:
            raise ValueError(f"Unknown events: {invalid}")
        return v
