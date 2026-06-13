import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_batch_agent_processes_all_invoices():
    from agents.batch_agent import run_batch
    invoice_ids = [str(uuid.uuid4()) for _ in range(3)]
    mock_db = AsyncMock()
    mock_provider = MagicMock()

    async def fake_extract(inv_id, db, provider):
        from models.invoice import InvoiceSchema
        return InvoiceSchema(vendor_name="Test Co", total_amount=100.0)

    with patch("agents.batch_agent._extract_single", side_effect=fake_extract):
        results = await run_batch(invoice_ids, mock_db, mock_provider)

    assert len(results["done"]) == 3
    assert len(results["failed"]) == 0
    assert results["total"] == 3
    assert results["success_count"] == 3
    assert results["failure_count"] == 0

@pytest.mark.asyncio
async def test_batch_agent_handles_partial_failure():
    from agents.batch_agent import run_batch
    invoice_ids = [str(uuid.uuid4()) for _ in range(3)]
    mock_db = AsyncMock()
    mock_provider = MagicMock()

    async def fake_extract_with_failure(inv_id, db, provider):
        if inv_id == invoice_ids[1]:
            raise ValueError("extraction failed")
        from models.invoice import InvoiceSchema
        return InvoiceSchema(vendor_name="Test Co", total_amount=100.0)

    with patch("agents.batch_agent._extract_single", side_effect=fake_extract_with_failure):
        results = await run_batch(invoice_ids, mock_db, mock_provider)

    assert len(results["done"]) == 2
    assert len(results["failed"]) == 1
    assert results["total"] == 3
    assert results["success_count"] == 2
    assert results["failure_count"] == 1
