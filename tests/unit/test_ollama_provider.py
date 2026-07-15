import pytest
from unittest.mock import patch, MagicMock
import json


def test_ollama_embed_text_returns_768_dims():
    fake_response = MagicMock()
    fake_response.json.return_value = {"embedding": [0.1] * 768}
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        from agents.providers.ollama_gemma import OllamaGemmaProvider
        provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")
        result = provider.embed_text(["test text"])
    assert len(result) == 1
    assert len(result[0]) == 768


def test_ollama_generate_returns_response():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "Paris"}
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        from agents.providers.ollama_gemma import OllamaGemmaProvider
        provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")
        result = provider.generate("What is the capital of France?")
    assert result == "Paris"


def test_ollama_generate_structured_returns_dict():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": '{"vendor_name": "Acme Corp"}'}
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        from agents.providers.ollama_gemma import OllamaGemmaProvider
        provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")

        class MockSchema:
            pass

        result = provider.generate_structured('Extract invoice fields', MockSchema)
    assert isinstance(result, dict)
    assert result.get("vendor_name") == "Acme Corp"
