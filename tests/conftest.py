import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")


@pytest.fixture(scope="session")
def invoice_pdf():
    p = FIXTURES_DIR / "invoice_1.pdf"
    assert p.exists(), f"Test fixture not found: {p}\nCopy invoice_1.pdf to tests/fixtures/"
    return p
