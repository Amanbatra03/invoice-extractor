from api.schemas.invoice import InvoiceOut, InvoiceUploadResponse
from api.schemas.job import JobOut
from api.schemas.webhook import WebhookIn, WebhookOut
import uuid
import datetime


def test_invoice_out_schema():
    inv = InvoiceOut(
        id=uuid.uuid4(), tenant_id=uuid.uuid4(), file_name="test.pdf",
        file_type="pdf", status="ready", sha256="abc", storage_path="path",
        created_at=datetime.datetime.utcnow(),
    )
    assert inv.file_type == "pdf"


def test_webhook_in_validates_events():
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        WebhookIn(url="https://example.com", events=[], secret="s")
