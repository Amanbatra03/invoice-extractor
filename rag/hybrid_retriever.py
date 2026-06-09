import pickle
from pathlib import Path

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.utils import load_config


def rrf_score(rank_bm25: int, rank_dense: int, k: int = 60) -> float:
    return 1 / (k + rank_bm25) + 1 / (k + rank_dense)


class HybridRetriever:
    def __init__(self, sha_key: str, base_dir: Path = Path(".")):
        cfg = load_config()
        self._num_results = cfg.NUM_RESULTS
        vectorstore_dir = base_dir / "vectorstore" / sha_key

        # Load BM25 index
        bm25_path = vectorstore_dir / "bm25.pkl"
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._texts = data["texts"]
        self._chunks = data["chunks"]

        # Load embeddings model for query encoding
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

        # Dense retrieval via ChromaDB
        query_embedding = self._embeddings.embed_query(query)
        dense_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        dense_docs = dense_results["documents"][0]  # list of text strings
        dense_metas = dense_results["metadatas"][0]  # list of metadata dicts

        dense_ranks: dict[int, int] = {}
        for rank, doc_text in enumerate(dense_docs):
            for i, text in enumerate(self._texts):
                if doc_text == text and i not in dense_ranks:
                    dense_ranks[i] = rank
                    break

        # RRF fusion
        all_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        fused = []
        for idx in all_indices:
            score = rrf_score(
                bm25_ranks.get(idx, n + 60),
                dense_ranks.get(idx, n + 60),
            )
            chunk = self._chunks[idx]
            fused.append({
                "text": self._texts[idx],
                "page": chunk.metadata.get("page", "?"),
                "score": score,
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: self._num_results]
