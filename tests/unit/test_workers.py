import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_extract_job_writes_extraction_to_db():
    from workers.extract_job import _run_async
    invoice_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    mock_schema = MagicMock()
    mock_schema.model_dump_json.return_value = '{"vendor_name": "Acme"}'
    mock_schema.model_dump.return_value = {"vendor_name": "Acme"}

    with patch("workers.extract_job.get_session_factory") as mock_sf:
        mock_db = AsyncMock()
        invoice_mock = MagicMock(id=uuid.UUID(invoice_id), tenant_id=uuid.uuid4(), file_type="pdf")
        mock_db.scalar = AsyncMock(side_effect=[invoice_mock, None])
        mock_sf.return_value.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("workers.extract_job.run_extraction", return_value=mock_schema):
            with patch("workers.extract_job.HybridRetriever"):
                with patch("workers.extract_job.get_provider"):
                    with patch("workers.extract_job.run_validation", return_value=MagicMock(issues=[])):
                        await _run_async(invoice_id, job_id)
        assert mock_db.add.called

def test_webhook_signs_payload():
    from workers.webhook_job import _build_signed_request
    payload = {"event": "extraction.completed", "tenant_id": "t1", "data": {}, "timestamp": 1000}
    headers = _build_signed_request(payload, "my_secret")
    assert "X-Signature" in headers
    assert headers["X-Signature"].startswith("sha256=")
