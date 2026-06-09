import re
from pathlib import Path
import box
import yaml


def load_config(path: Path | str | None = None) -> box.Box:
    if path is None:
        path = Path(__file__).parent.parent / "config.yml"
    with open(path, "r", encoding="utf8") as f:
        return box.Box(yaml.safe_load(f))


def extract_json_from_text(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None
