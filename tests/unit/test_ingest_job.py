import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_ingest_splits_pdf_into_chunks():
    from workers.ingest_job import _chunk_pdf_bytes
    fake_pdf_text = ["Page one content about an invoice from Acme Corp.", "Page two with line items."]
    with patch("workers.ingest_job._extract_page_texts", return_value=fake_pdf_text):
        chunks = _chunk_pdf_bytes(b"%PDF fake", chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all("text" in c and "page" in c for c in chunks)

@pytest.mark.asyncio
async def test_ingest_stores_chunks_in_db():
    from workers.ingest_job import _store_chunks
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_provider.embed_text.return_value = [[0.1] * 768 for _ in range(3)]
    invoice_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    chunks = [{"text": f"chunk {i}", "page": 1} for i in range(3)]
    await _store_chunks(chunks, invoice_id, tenant_id, mock_db, mock_provider)
    assert mock_db.add.call_count == 3
    assert mock_db.commit.called
