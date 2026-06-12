import asyncio
import re
import uuid
from typing import Any

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

    async def _load_corpus(self) -> list[dict]:
        if self._corpus is None:
            result = await self._db.execute(
                text(
                    "SELECT id, chunk_text, page_num FROM invoice_chunks "
                    "WHERE invoice_id = :inv_id ORDER BY page_num"
                ),
                {"inv_id": str(self._invoice_id)},
            )
            # result.mappings() is synchronous in real SQLAlchemy, but
            # AsyncMock auto-wraps child attributes as coroutines in tests.
            mappings_result = result.mappings()
            if asyncio.iscoroutine(mappings_result):
                mappings_result = await mappings_result
            rows = mappings_result.all()
            if asyncio.iscoroutine(rows):
                rows = await rows
            self._corpus = [
                {"id": str(r["id"]), "chunk_text": r["chunk_text"], "page_num": r["page_num"]}
                for r in rows
            ]
        return self._corpus

    def _bm25_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        tokenized = [_tokenize(c["chunk_text"]) for c in corpus]
        bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[""]])
        scores = bm25.get_scores(_tokenize(query))
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
        query_embedding = self._provider.embed_text([query])[0]
        result = await self._db.execute(
            text(
                "SELECT id, chunk_text, page_num, "
                "1 - (embedding <=> CAST(:emb AS vector)) AS similarity "
                "FROM invoice_chunks "
                "WHERE invoice_id = :inv_id "
                "ORDER BY embedding <=> CAST(:emb AS vector) "
                "LIMIT :n"
            ),
            {
                "emb": str(query_embedding),
                "inv_id": str(self._invoice_id),
                "n": n,
            },
        )
        rows = result.mappings().all()
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

        # Support both sync and async versions of _bm25_retrieve/_dense_retrieve
        # (the async variant is for the real implementation; sync for testability)
        bm25_raw = self._bm25_retrieve(corpus, query, n)
        bm25_result = await bm25_raw if asyncio.iscoroutine(bm25_raw) else bm25_raw

        dense_raw = self._dense_retrieve(corpus, query, n)
        dense_results = await dense_raw if asyncio.iscoroutine(dense_raw) else dense_raw

        bm25_results: list[dict] = bm25_result  # type: ignore[assignment]

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
