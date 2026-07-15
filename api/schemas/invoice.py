import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class InvoiceOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    file_name: str
    file_type: str
    status: str
    sha256: str
    storage_path: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InvoiceUploadResponse(BaseModel):
    invoice_id: uuid.UUID
    job_id: uuid.UUID
    status: str


class InvoiceListResponse(BaseModel):
    items: list[InvoiceOut]
    total: int
    page: int
    limit: int
