import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ExtractionOut(BaseModel):
    id: uuid.UUID
    invoice_id: uuid.UUID
    schema_json: dict
    model_used: str
    validated: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ValidationResult(BaseModel):
    passed: bool
    issues: list[dict]
