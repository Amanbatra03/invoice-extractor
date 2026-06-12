import pytest
from pathlib import Path

from eval.generate_dataset import generate_dataset
from eval.scoring import field_match, score_invoice
from models.invoice import InvoiceSchema
from rag.validator import validate_invoice


def test_generate_dataset_creates_labeled_pairs(tmp_path):
    pairs = generate_dataset(tmp_path, n=3, seed=7)
    assert len(pairs) == 3
    for pdf_path, truth_path in pairs:
        assert pdf_path.exists() and pdf_path.suffix == ".pdf"
        truth = InvoiceSchema.model_validate_json(truth_path.read_text(encoding="utf8"))
        assert truth.vendor_name and truth.invoice_number
        assert validate_invoice(truth) == []


def test_generate_dataset_deterministic(tmp_path):
    a = generate_dataset(tmp_path / "a", n=2, seed=42)
    b = generate_dataset(tmp_path / "b", n=2, seed=42)
    truth_a = a[0][1].read_text(encoding="utf8")
    truth_b = b[0][1].read_text(encoding="utf8")
    assert truth_a == truth_b


def test_field_match_strings_case_insensitive():
    assert field_match("ACME Corp", "acme corp") is True
    assert field_match("ACME Corp", "Beta Ltd") is False


def test_field_match_numbers_tolerant():
    assert field_match(212.09, 212.10) is True
    assert field_match(212.09, 250.0) is False


def test_field_match_none_handling():
    assert field_match(None, None) is True
    assert field_match(None, "x") is False


def test_score_invoice_perfect():
    truth = InvoiceSchema(vendor_name="A", invoice_number="1", total_amount=10.0, currency="USD")
    scores = score_invoice(truth, truth)
    assert all(v == 1.0 for v in scores.values())
