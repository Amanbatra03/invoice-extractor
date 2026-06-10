import json
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from rag.utils import load_config


def rrf_score(rank_bm25: int, rank_dense: int, k: int = 60) -> float:
    return 1 / (k + rank_bm25) + 1 / (k + rank_dense)


class HybridRetriever:
    def __init__(self, sha_key: str, base_dir: Path = Path(".")):
        cfg = load_config()
        self._num_results = cfg.NUM_RESULTS
        vectorstore_dir = base_dir / "vectorstore" / sha_key

        # Load BM25 corpus from JSON and rebuild the index
        index_path = vectorstore_dir / "bm25.json"
        data = json.loads(index_path.read_text(encoding="utf8"))
        self._texts = data["texts"]
        self._pages = data["pages"]
        self._bm25 = BM25Okapi([t.split() for t in self._texts])

        # Load embeddings model for query encoding (heavy import deferred)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=cfg.EMBEDDINGS,
            model_kwargs={"device": cfg.DEVICE},
            encode_kwargs={"normalize_embeddings": cfg.NORMALIZE_EMBEDDINGS},
        )

        # Load ChromaDB collection (chromadb >= 0.4 PersistentClient API)
        client = chromadb.PersistentClient(path=str(vectorstore_dir))
        self._collection = client.get_collection(name=f"invoice_{sha_key}")

    def retrieve(self, query: str) -> list[dict]:
        n = min(self._num_results * 3, len(self._texts))

        # BM25 ranking
        tokenized_query = query.split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        sorted_bm25_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )
        bm25_ranks = {idx: rank for rank, idx in enumerate(sorted_bm25_idx[:n])}

        # Dense retrieval via ChromaDB; ids are "chunk_<i>" so they map back to corpus indices
        query_embedding = self._embeddings.embed_query(query)
        dense_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        dense_ranks = {
            int(chunk_id.rsplit("_", 1)[1]): rank
            for rank, chunk_id in enumerate(dense_results["ids"][0])
        }

        # RRF fusion
        all_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        fused = []
        for idx in all_indices:
            score = rrf_score(
                bm25_ranks.get(idx, n + 60),
                dense_ranks.get(idx, n + 60),
            )
            fused.append({
                "text": self._texts[idx],
                "page": self._pages[idx],
                "score": score,
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: self._num_results]
