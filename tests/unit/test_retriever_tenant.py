import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


def _rows(tenant=True):
    base = [
        {"id": uuid.uuid4(), "chunk_text": "Total due $500 from Acme Corp", "page_num": 1},
        {"id": uuid.uuid4(), "chunk_text": "Total due $900 from Globex Inc", "page_num": 1},
    ]
    if tenant:
        for i, r in enumerate(base):
            r["invoice_id"] = uuid.uuid4()
            r["file_name"] = ["acme.pdf", "globex.pdf"][i]
    return base


@pytest.mark.asyncio
async def test_tenant_mode_filters_by_tenant_and_returns_file_names():
    from agents.retriever import HybridRetriever
    tenant_id = uuid.uuid4()
    rows = _rows()
    queries = []

    async def fake_execute(query, params):
        queries.append((str(query), params))
        if "embedding" in str(query):
            return FakeResult([{**r, "similarity": 0.9} for r in rows])
        return FakeResult(rows)

    db = MagicMock()
    db.execute = AsyncMock(side_effect=fake_execute)
    provider = MagicMock()
    provider.embed_text = MagicMock(return_value=[[0.0] * 768])

    r = HybridRetriever(invoice_id=None, db=db, provider=provider, num_results=2, tenant_id=tenant_id)
    out = await r.retrieve("total due")

    assert len(out) == 2
    assert all("file_name" in c and "invoice_id" in c for c in out)
    corpus_sql, corpus_params = queries[0]
    assert "tenant_id" in corpus_sql
    assert "file_type" in corpus_sql  # image placeholders excluded
    assert corpus_params["tid"] == str(tenant_id)


@pytest.mark.asyncio
async def test_single_invoice_mode_unchanged():
    from agents.retriever import HybridRetriever
    rows = _rows(tenant=False)
    queries = []

    async def fake_execute(query, params):
        queries.append((str(query), params))
        if "embedding" in str(query):
            return FakeResult([{**r, "similarity": 0.9} for r in rows])
        return FakeResult(rows)

    db = MagicMock()
    db.execute = AsyncMock(side_effect=fake_execute)
    provider = MagicMock()
    provider.embed_text = MagicMock(return_value=[[0.0] * 768])
    inv_id = uuid.uuid4()

    r = HybridRetriever(invoice_id=inv_id, db=db, provider=provider, num_results=2)
    out = await r.retrieve("total due")
    assert len(out) == 2
    assert "inv_id" in queries[0][1]


def test_requires_invoice_or_tenant():
    from agents.retriever import HybridRetriever
    with pytest.raises(ValueError):
        HybridRetriever(invoice_id=None, db=MagicMock(), provider=MagicMock(), tenant_id=None)
