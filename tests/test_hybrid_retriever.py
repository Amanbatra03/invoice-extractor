import pytest
from pathlib import Path

from rag.hybrid_retriever import rrf_score, HybridRetriever
from ingest import ingest_pdf


def test_rrf_score_zero_ranks():
    score = rrf_score(rank_bm25=0, rank_dense=0, k=60)
    assert abs(score - 2 / 60) < 1e-9


def test_rrf_score_lower_rank_gives_higher_score():
    high = rrf_score(rank_bm25=0, rank_dense=0)
    low = rrf_score(rank_bm25=10, rank_dense=10)
    assert high > low


def test_rrf_score_custom_k():
    score = rrf_score(rank_bm25=0, rank_dense=0, k=10)
    assert abs(score - 2 / 10) < 1e-9


@pytest.mark.slow
def test_retrieve_returns_sorted_results(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    retriever = HybridRetriever(sha_key, base_dir=tmp_path)
    results = retriever.retrieve("What is the total amount due?")
    assert len(results) > 0
    assert all(k in results[0] for k in ("text", "page", "score"))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_retrieve_returns_at_most_num_results(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    retriever = HybridRetriever(sha_key, base_dir=tmp_path)
    results = retriever.retrieve("invoice number")
    assert len(results) <= 4  # NUM_RESULTS in config


@pytest.mark.slow
def test_all_chunks_returns_full_corpus_in_order(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    retriever = HybridRetriever(sha_key, base_dir=tmp_path)
    chunks = retriever.all_chunks()
    assert len(chunks) >= 1
    assert all(set(c) >= {"text", "page"} for c in chunks)
    full_text = " ".join(c["text"] for c in chunks)
    assert "212,09" in full_text or "212.09" in full_text  # the gross total survives
