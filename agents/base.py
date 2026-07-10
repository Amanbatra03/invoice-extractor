from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_image(self, image_path: Path) -> list[float]:
        ...

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        ...

    def generate_structured(self, prompt: str, schema: type) -> dict:
        ...

    def generate_with_image(self, prompt: str, image_path: Path, system: str | None = None) -> str:
        ...


def get_provider() -> LLMProvider:
    from api.config import get_settings
    settings = get_settings()
    if settings.LLM_PROVIDER == "ollama_gemma":
        from agents.providers.ollama_gemma import OllamaGemmaProvider
        return OllamaGemmaProvider()
    from agents.providers.gemini import GeminiProvider
    return GeminiProvider()
