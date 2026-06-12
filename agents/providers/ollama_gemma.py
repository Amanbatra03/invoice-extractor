import base64
import json
import re
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()


class OllamaGemmaProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        if base_url is None or model is None:
            from api.config import get_settings
            settings = get_settings()
            self._base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
            self._model = model or settings.GEMMA_MODEL
        else:
            self._base_url = base_url.rstrip("/")
            self._model = model

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        with httpx.Client(timeout=30) as client:
            for text in texts:
                resp = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data["embedding"])
        return embeddings

    def embed_image(self, image_path: Path) -> list[float]:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        prompt = "Describe and embed this invoice image for semantic search."
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": prompt, "images": [img_b64]},
            )
            resp.raise_for_status()
            data = resp.json()
        return data["embedding"]

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if "images" in kwargs:
            payload["images"] = kwargs["images"]
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
        return resp.json().get("response", "")

    def generate_structured(self, prompt: str, schema: type) -> dict:
        structured_prompt = (
            f"{prompt}\n\nRespond ONLY with valid JSON matching this structure. "
            "Do not include any explanation or markdown formatting."
        )
        raw = self.generate(structured_prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        log.warning("ollama_structured_parse_failed", raw=raw[:200])
        return {}
