from __future__ import annotations

from pathlib import Path
from typing import Any

from vocabpruner.storage.paths import hf_models_dir


def resolve_gguf(model: str | Path) -> Path | None:
    path = Path(model)
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path
    if path.is_dir():
        matches = sorted(path.glob("*.gguf"))
    else:
        repo = hf_models_dir() / f"models--{str(model).replace('/', '--')}"
        revision = (repo / "refs" / "main").read_text().strip() if (repo / "refs" / "main").is_file() else ""
        snapshots = [repo / "snapshots" / revision] if revision else sorted((repo / "snapshots").glob("*"))
        matches = sorted(file for snapshot in snapshots if snapshot.is_dir() for file in snapshot.glob("*.gguf"))
    if len(matches) > 1:
        raise ValueError(f"Multiple GGUF files found for {model}; pass the exact file path.")
    return matches[0] if matches else None


def read_gguf(model: str | Path):
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("GGUF support requires the gguf package. Install project dependencies first.") from exc
    return GGUFReader(model)


def gguf_field(reader: Any, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    return default if field is None else field.contents()
