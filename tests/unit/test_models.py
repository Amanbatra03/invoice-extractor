import uuid
from db.models import Tenant, User, Invoice, InvoiceChunk, Extraction, Job, Webhook, WebhookDelivery, AuditLog, ApiKey, LlmUsage

def test_tenant_has_required_columns():
    cols = {c.name for c in Tenant.__table__.columns}
    assert {"id", "name", "plan", "created_at"}.issubset(cols)

def test_invoice_chunk_has_vector_column():
    cols = {c.name for c in InvoiceChunk.__table__.columns}
    assert "embedding" in cols

def test_user_role_choices():
    col = User.__table__.columns["role"]
    assert col is not None

def test_all_tables_have_tenant_id():
    tables_needing_tenant = [Invoice, InvoiceChunk, Extraction, Job, Webhook, AuditLog, ApiKey, LlmUsage]
    for model in tables_needing_tenant:
        cols = {c.name for c in model.__table__.columns}
        assert "tenant_id" in cols, f"{model.__name__} missing tenant_id"
