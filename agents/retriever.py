import asyncio
import re
import uuid

from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import LLMProvider


def _tokenize(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def _rrf_score(rank_a: int, rank_b: int, k: int = 60) -> float:
    return 1 / (k + rank_a) + 1 / (k + rank_b)


class HybridRetriever:
    def __init__(
        self,
        invoice_id: uuid.UUID | None = None,
        db: AsyncSession = None,
        provider: LLMProvider = None,
        num_results: int = 4,
        tenant_id: uuid.UUID | None = None,
    ):
        if invoice_id is None and tenant_id is None:
            raise ValueError("HybridRetriever requires an invoice_id or a tenant_id")
        self._invoice_id = invoice_id
        self._tenant_id = tenant_id
        self._db = db
        self._provider = provider
        self._num_results = num_results
        self._corpus: list[dict] | None = None
        self._bm25: BM25Okapi | None = None

    async def _unwrap_rows(self, result) -> list:
        mappings = result.mappings()
        if asyncio.iscoroutine(mappings):
            mappings = await mappings
        rows = mappings.all()
        if asyncio.iscoroutine(rows):
            rows = await rows
        return list(rows)

    async def _load_corpus(self) -> list[dict]:
        if self._corpus is None:
            if self._invoice_id is not None:
                result = await self._db.execute(
                    text(
                        "SELECT id, chunk_text, page_num FROM invoice_chunks "
                        "WHERE invoice_id = :inv_id ORDER BY page_num"
                    ),
                    {"inv_id": str(self._invoice_id)},
                )
            else:
                result = await self._db.execute(
                    text(
                        "SELECT ic.id, ic.chunk_text, ic.page_num, ic.invoice_id, i.file_name "
                        "FROM invoice_chunks ic JOIN invoices i ON i.id = ic.invoice_id "
                        "WHERE ic.tenant_id = :tid AND i.file_type != 'image' "
                        "ORDER BY i.file_name, ic.page_num"
                    ),
                    {"tid": str(self._tenant_id)},
                )
            rows = await self._unwrap_rows(result)
            self._corpus = []
            for r in rows:
                entry = {"id": str(r["id"]), "chunk_text": r["chunk_text"], "page_num": r["page_num"]}
                if "file_name" in r.keys():
                    entry["invoice_id"] = str(r["invoice_id"])
                    entry["file_name"] = r["file_name"]
                self._corpus.append(entry)
        return self._corpus

    def _bm25_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        if self._bm25 is None:
            tokenized = [_tokenize(c["chunk_text"]) for c in corpus]
            self._bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[""]])
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {**corpus[i], "score": float(scores[i])}
            for i in ranked[:n]
        ]

    async def _dense_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        query_embedding = await asyncio.to_thread(self._provider.embed_text, [query])
        query_embedding = query_embedding[0]
        if self._invoice_id is not None:
            sql = (
                "WITH q AS (SELECT CAST(:emb AS vector) AS vec) "
                "SELECT ic.id, ic.chunk_text, ic.page_num, "
                "1 - (ic.embedding <=> q.vec) AS similarity "
                "FROM invoice_chunks ic, q "
                "WHERE ic.invoice_id = :inv_id "
                "ORDER BY ic.embedding <=> q.vec "
                "LIMIT :n"
            )
            params = {"emb": str(query_embedding), "inv_id": str(self._invoice_id), "n": n}
        else:
            sql = (
                "WITH q AS (SELECT CAST(:emb AS vector) AS vec) "
                "SELECT ic.id, ic.chunk_text, ic.page_num, ic.invoice_id, i.file_name, "
                "1 - (ic.embedding <=> q.vec) AS similarity "
                "FROM invoice_chunks ic JOIN invoices i ON i.id = ic.invoice_id, q "
                "WHERE ic.tenant_id = :tid AND i.file_type != 'image' "
                "ORDER BY ic.embedding <=> q.vec "
                "LIMIT :n"
            )
            params = {"emb": str(query_embedding), "tid": str(self._tenant_id), "n": n}
        result = await self._db.execute(text(sql), params)
        rows = await self._unwrap_rows(result)
        out = []
        for r in rows:
            entry = {
                "id": str(r["id"]),
                "chunk_text": r["chunk_text"],
                "page_num": r["page_num"],
                "score": float(r["similarity"]),
            }
            if "file_name" in r.keys():
                entry["invoice_id"] = str(r["invoice_id"])
                entry["file_name"] = r["file_name"]
            out.append(entry)
        return out

    async def retrieve(self, query: str) -> list[dict]:
        corpus = await self._load_corpus()
        if not corpus:
            return []
        n = min(self._num_results * 3, len(corpus))

        bm25_results = self._bm25_retrieve(corpus, query, n)
        dense_results = await self._dense_retrieve(corpus, query, n)

        bm25_ranks = {r["id"]: rank for rank, r in enumerate(bm25_results)}
        dense_ranks = {r["id"]: rank for rank, r in enumerate(dense_results)}
        all_ids = set(bm25_ranks) | set(dense_ranks)

        # Build id→chunk lookup from both result sets
        chunk_by_id: dict[str, dict] = {}
        for r in bm25_results + dense_results:
            chunk_by_id[r["id"]] = r

        sentinel = n + 60
        fused = []
        for cid in all_ids:
            c = chunk_by_id[cid]
            entry = {
                "text": c["chunk_text"],
                "page": c["page_num"],
                "score": _rrf_score(
                    bm25_ranks.get(cid, sentinel),
                    dense_ranks.get(cid, sentinel),
                ),
            }
            if "file_name" in c:
                entry["invoice_id"] = c["invoice_id"]
                entry["file_name"] = c["file_name"]
            fused.append(entry)
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: self._num_results]

    async def all_chunks(self) -> list[dict]:
        corpus = await self._load_corpus()
        return [{"text": c["chunk_text"], "page": c["page_num"]} for c in corpus]
