"""Run extraction over the synthetic dataset and report per-field accuracy.

Requires Ollama running with the configured model. Usage:
    python -m eval.run_eval [--n 12]
"""
import argparse
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from eval.generate_dataset import generate_dataset
from eval.scoring import score_invoice, _HEADER_FIELDS
from ingest import ingest_pdf
from models.invoice import InvoiceSchema
from rag.extractor import extract_invoice, ExtractionError
from rag.hybrid_retriever import HybridRetriever
from rag.llm import get_ollama_llm
from rag.utils import load_config


def main(n: int = 12) -> Path:
    cfg = load_config()
    dataset_dir = Path(__file__).parent / "dataset"
    pairs = generate_dataset(dataset_dir, n=n)
    work = Path(tempfile.mkdtemp(prefix="eval_"))
    llm = get_ollama_llm(cfg.LLM, format_schema=InvoiceSchema.model_json_schema(),
                         num_ctx=int(cfg.NUM_CTX))

    totals: dict[str, list[float]] = defaultdict(list)
    failures = 0
    for pdf_path, truth_path in pairs:
        truth = InvoiceSchema.model_validate_json(truth_path.read_text(encoding="utf8"))
        sha = ingest_pdf(pdf_path, base_dir=work)
        retriever = HybridRetriever(sha, base_dir=work)
        try:
            predicted = extract_invoice(retriever, llm)
        except ExtractionError:
            failures += 1
            predicted = InvoiceSchema()
        for field, score in score_invoice(truth, predicted).items():
            totals[field].append(score)
        print(f"  {pdf_path.name}: done")

    lines = ["# Extraction Eval Results", "",
             f"Model: `{cfg.LLM}` (schema-constrained, whole-document) — "
             f"{n} synthetic invoices, {failures} hard failures", "",
             "| Field | Accuracy |", "|---|---|"]
    for field in _HEADER_FIELDS + ["line_items"]:
        if field in totals:
            acc = sum(totals[field]) / len(totals[field])
            lines.append(f"| {field} | {acc:.0%} |")
    overall = sum(sum(v) for v in totals.values()) / sum(len(v) for v in totals.values())
    lines += ["", f"**Overall field accuracy: {overall:.0%}**"]

    out = Path(__file__).parent / "results.md"
    out.write_text("\n".join(lines), encoding="utf8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {out}\noverall: {overall:.0%}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    main(parser.parse_args().n)
