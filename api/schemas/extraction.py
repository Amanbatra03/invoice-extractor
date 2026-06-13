import uuid
from datetime import datetime
from pydantic import BaseModel


class ExtractionOut(BaseModel):
    id: uuid.UUID
    invoice_id: uuid.UUID
    schema_json: dict
    model_used: str
    validated: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ValidationResult(BaseModel):
    passed: bool
    issues: list[dict]
