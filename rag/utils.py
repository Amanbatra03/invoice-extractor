import re
from functools import lru_cache
from pathlib import Path
import box
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yml"


@lru_cache(maxsize=1)
def _default_config() -> box.Box:
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf8") as f:
        return box.Box(yaml.safe_load(f))


def load_config(path: Path | str | None = None) -> box.Box:
    # Default config is a shared singleton so runtime overrides (e.g. from the
    # app sidebar) are visible to every module that calls load_config().
    if path is None:
        return _default_config()
    with open(path, "r", encoding="utf8") as f:
        return box.Box(yaml.safe_load(f))


def extract_json_from_text(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None
