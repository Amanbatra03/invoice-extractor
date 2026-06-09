import pytest
from pathlib import Path
from rag.utils import extract_json_from_text, load_config


def test_extract_json_simple():
    text = 'Result: {"key": "value", "num": 42} done.'
    result = extract_json_from_text(text)
    assert result == '{"key": "value", "num": 42}'


def test_extract_json_multiline():
    text = 'Answer:\n{\n  "vendor": "ACME",\n  "total": 100.0\n}\ndone.'
    result = extract_json_from_text(text)
    assert result is not None
    assert '"vendor": "ACME"' in result


def test_extract_json_no_json():
    result = extract_json_from_text("No JSON here at all.")
    assert result is None


def test_load_config_returns_box(tmp_path):
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("CHUNK_SIZE: 800\nNUM_RESULTS: 4\n")
    cfg = load_config(cfg_file)
    assert cfg.CHUNK_SIZE == 800
    assert cfg.NUM_RESULTS == 4
