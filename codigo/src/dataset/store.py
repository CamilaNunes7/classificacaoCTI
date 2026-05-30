import json
from pathlib import Path
from typing import List

from src.dataset.schema import Indicator


def save_dataset(indicators: List[Indicator], path: Path, metadata: dict = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "1.0",
        **(metadata or {}),
        "total_indicators": len(indicators),
        "indicators": [ind.to_dict() for ind in indicators],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset(path: Path) -> List[Indicator]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Indicator.from_dict(d) for d in data.get("indicators", [])]


def load_dataset_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "indicators"}


def dataset_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 100
