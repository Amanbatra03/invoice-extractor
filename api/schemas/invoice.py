import uuid
from datetime import datetime
from pydantic import BaseModel


class InvoiceOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    file_name: str
    file_type: str
    status: str
    sha256: str
    storage_path: str
    created_at: datetime

    class Config:
        from_attributes = True


class InvoiceUploadResponse(BaseModel):
    invoice_id: uuid.UUID
    job_id: uuid.UUID
    status: str


class InvoiceListResponse(BaseModel):
    items: list[InvoiceOut]
    total: int
    page: int
    limit: int
