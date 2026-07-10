import os
from unittest.mock import MagicMock, patch

import httpx

_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "REDIS_URL": "redis://localhost:6379",
    "SUPABASE_URL": "https://fake.supabase.co",
    "SUPABASE_ANON_KEY": "fake_anon",
    "SUPABASE_SERVICE_KEY": "fake_service",
    "SUPABASE_JWT_SECRET": "test_secret",
    "GOOGLE_API_KEY": "fake_google_key",
}


def test_provider_protocol_includes_vision():
    from agents.base import LLMProvider
    assert hasattr(LLMProvider, "generate_with_image")


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "The total is $42"}


def _fake_client_factory(captured, post_exc=None):
    class _FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            if post_exc:
                raise post_exc
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    return _FakeClient


def test_ollama_generate_with_image_sends_image_payload(tmp_path):
    from agents.providers.ollama_gemma import OllamaGemmaProvider
    img = tmp_path / "invoice.png"
    img.write_bytes(b"fake-png-bytes")
    provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")
    captured = {}
    with patch("agents.providers.ollama_gemma.httpx.Client", _fake_client_factory(captured)):
        out = provider.generate_with_image("what is the total?", img)
    assert out == "The total is $42"
    assert captured["json"]["images"], "expected base64 image in payload"


def test_ollama_generate_with_image_degrades_gracefully(tmp_path):
    from agents.providers.ollama_gemma import OllamaGemmaProvider
    img = tmp_path / "invoice.png"
    img.write_bytes(b"fake-png-bytes")
    provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")
    exc = httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())
    with patch("agents.providers.ollama_gemma.httpx.Client", _fake_client_factory({}, post_exc=exc)):
        out = provider.generate_with_image("what is the total?", img)
    assert "image" in out.lower()  # explains, does not raise


def test_gemini_generate_with_image_calls_multimodal(tmp_path):
    img = tmp_path / "invoice.png"
    img.write_bytes(b"fake-png-bytes")
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        with patch("agents.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = MagicMock(text="Total: $99")
            mock_client_cls.return_value = mock_client
            from agents.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            with patch("PIL.Image.open", return_value=MagicMock()):
                out = provider.generate_with_image("what is the total?", img)
    assert out == "Total: $99"
    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 2  # prompt + image
