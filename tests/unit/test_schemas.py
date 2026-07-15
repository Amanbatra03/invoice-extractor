from api.schemas.invoice import InvoiceOut
from api.schemas.webhook import WebhookIn
import uuid
import datetime


def test_invoice_out_schema():
    inv = InvoiceOut(
        id=uuid.uuid4(), tenant_id=uuid.uuid4(), file_name="test.pdf",
        file_type="pdf", status="ready", sha256="abc", storage_path="path",
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    assert inv.file_type == "pdf"


def test_webhook_in_validates_events():
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        WebhookIn(url="https://example.com", events=[], secret="s")
