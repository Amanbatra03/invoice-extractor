import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_retriever_returns_top_n_chunks():
    from agents.retriever import HybridRetriever
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_provider.embed_text.return_value = [[0.1] * 768]

    chunks = [
        {"id": str(uuid.uuid4()), "chunk_text": f"chunk {i}", "page_num": 1, "score": 0.9 - i * 0.1}
        for i in range(5)
    ]
    mock_db.execute = AsyncMock()
    mock_db.execute.return_value.mappings.return_value.all.return_value = [
        {"chunk_text": c["chunk_text"], "page_num": c["page_num"], "id": c["id"]}
        for c in chunks[:4]
    ]

    retriever = HybridRetriever(
        invoice_id=uuid.uuid4(),
        db=mock_db,
        provider=mock_provider,
        num_results=4,
    )
    with patch.object(retriever, "_bm25_retrieve", return_value=chunks[:4]):
        with patch.object(retriever, "_dense_retrieve", return_value=chunks[:4]):
            results = await retriever.retrieve("what is the total")
    assert len(results) <= 4
    assert all("text" in r and "page" in r and "score" in r for r in results)

@pytest.mark.asyncio
async def test_all_chunks_returns_full_corpus():
    from agents.retriever import HybridRetriever
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.execute.return_value.mappings.return_value.all.return_value = [
        {"chunk_text": f"text {i}", "page_num": i + 1, "id": str(uuid.uuid4())}
        for i in range(10)
    ]
    retriever = HybridRetriever(
        invoice_id=uuid.uuid4(),
        db=mock_db,
        provider=mock_provider,
        num_results=4,
    )
    results = await retriever.all_chunks()
    assert len(results) == 10
