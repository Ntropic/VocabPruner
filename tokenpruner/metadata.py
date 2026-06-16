from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


MODEL_METADATA_FILE = "tokenpruner_model.json"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [ _to_jsonable(v) for v in obj ]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def write_prune_metadata(pruned_model: str | Path, source_model: str, score_config: Any) -> Path:
    path = Path(pruned_model) / MODEL_METADATA_FILE
    payload: dict[str, Any] = {
        "source_model": str(source_model),
        "pruned_model": str(Path(pruned_model).absolute()),
    }
    if is_dataclass(score_config):
        payload["score_config"] = _to_jsonable(asdict(score_config))
    else:
        payload["score_config"] = _to_jsonable(dict(score_config)) if isinstance(score_config, dict) else {}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
