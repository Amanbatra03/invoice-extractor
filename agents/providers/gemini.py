import json
from pathlib import Path

import structlog
from google import genai
from google.genai import types

from api.config import get_settings

log = structlog.get_logger()


class GeminiProvider:
    def __init__(self):
        settings = get_settings()
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model = settings.GEMINI_MODEL
        self._embed_model = settings.GEMINI_EMBEDDING_MODEL

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        embeddings = []
        for text in texts:
            result = self._client.models.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_image(self, image_path: Path) -> list[float]:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        result = self._client.models.embed_content(
            model=self._embed_model,
            contents=img,
        )
        return result.embeddings[0].values

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
        ) if system else None
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def generate_with_image(self, prompt: str, image_path: Path, system: str | None = None) -> str:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, img],
            config=config,
        )
        return response.text or ""

    def generate_structured(self, prompt: str, schema: type) -> dict:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = response.text or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {}
