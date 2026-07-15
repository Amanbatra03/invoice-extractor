def test_alert_model_columns():
    from db.models import Alert
    cols = {c.name for c in Alert.__table__.columns}
    expected = {
        "id", "severity", "source", "event", "detail", "context",
        "fingerprint", "delivery_status", "delivery_attempts",
        "last_error", "delivered_at", "created_at",
    }
    assert expected <= cols
    assert Alert.__tablename__ == "alerts"


def test_alert_model_has_no_tenant_fk():
    from db.models import Alert
    assert "tenant_id" not in {c.name for c in Alert.__table__.columns}
