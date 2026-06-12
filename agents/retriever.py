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
        invoice_id: uuid.UUID,
        db: AsyncSession,
        provider: LLMProvider,
        num_results: int = 4,
    ):
        self._invoice_id = invoice_id
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
            result = await self._db.execute(
                text(
                    "SELECT id, chunk_text, page_num FROM invoice_chunks "
                    "WHERE invoice_id = :inv_id ORDER BY page_num"
                ),
                {"inv_id": str(self._invoice_id)},
            )
            rows = await self._unwrap_rows(result)
            self._corpus = [
                {"id": str(r["id"]), "chunk_text": r["chunk_text"], "page_num": r["page_num"]}
                for r in rows
            ]
        return self._corpus

    def _bm25_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        if self._bm25 is None:
            tokenized = [_tokenize(c["chunk_text"]) for c in corpus]
            self._bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[""]])
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {
                "id": corpus[i]["id"],
                "chunk_text": corpus[i]["chunk_text"],
                "page_num": corpus[i]["page_num"],
                "score": float(scores[i]),
            }
            for i in ranked[:n]
        ]

    async def _dense_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        query_embedding = await asyncio.to_thread(self._provider.embed_text, [query])
        query_embedding = query_embedding[0]
        result = await self._db.execute(
            text(
                "WITH q AS (SELECT CAST(:emb AS vector) AS vec) "
                "SELECT ic.id, ic.chunk_text, ic.page_num, "
                "1 - (ic.embedding <=> q.vec) AS similarity "
                "FROM invoice_chunks ic, q "
                "WHERE ic.invoice_id = :inv_id "
                "ORDER BY ic.embedding <=> q.vec "
                "LIMIT :n"
            ),
            {
                "emb": str(query_embedding),
                "inv_id": str(self._invoice_id),
                "n": n,
            },
        )
        rows = await self._unwrap_rows(result)
        return [
            {
                "id": str(r["id"]),
                "chunk_text": r["chunk_text"],
                "page_num": r["page_num"],
                "score": float(r["similarity"]),
            }
            for r in rows
        ]

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
        fused = [
            {
                "text": chunk_by_id[cid]["chunk_text"],
                "page": chunk_by_id[cid]["page_num"],
                "score": _rrf_score(
                    bm25_ranks.get(cid, sentinel),
                    dense_ranks.get(cid, sentinel),
                ),
            }
            for cid in all_ids
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: self._num_results]

    async def all_chunks(self) -> list[dict]:
        corpus = await self._load_corpus()
        return [{"text": c["chunk_text"], "page": c["page_num"]} for c in corpus]
