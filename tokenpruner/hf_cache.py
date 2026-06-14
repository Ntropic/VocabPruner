from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenpruner.paths import HF_MODELS_DIR

HF_DOWNLOAD_MARKER = "tokenpruner_hf_download.json"
HF_MARKER_SCHEMA = 1
ANALYSIS_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.txt",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
)
CHECKPOINT_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
)


@dataclass(frozen=True)
class CachedCheckpointStatus:
    complete: bool
    kind: str
    marker_path: Path | None
    snapshot_path: Path | None


def repo_cache_dir(model: str) -> Path | None:
    if Path(model).exists():
        return None
    return HF_MODELS_DIR / f"models--{model.replace('/', '--')}"


def download_marker_path(model: str) -> Path | None:
    cache_dir = repo_cache_dir(model)
    if cache_dir is None:
        return None
    return cache_dir / HF_DOWNLOAD_MARKER


def read_download_marker(model: str) -> dict[str, Any] | None:
    marker_path = download_marker_path(model)
    if marker_path is None or not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(marker, dict) or marker.get("schema") != HF_MARKER_SCHEMA:
        return None
    return marker


def mark_analysis_download(model: str, trust_remote_code: bool) -> None:
    mark_download(
        model,
        kind="analysis",
        trust_remote_code=trust_remote_code,
        allow_patterns=ANALYSIS_ALLOW_PATTERNS + (("*.py",) if trust_remote_code else ()),
    )


def mark_checkpoint_download(model: str, trust_remote_code: bool, snapshot_path: Path) -> None:
    mark_download(
        model,
        kind="checkpoint",
        trust_remote_code=trust_remote_code,
        allow_patterns=CHECKPOINT_ALLOW_PATTERNS
        + (("*.py",) if trust_remote_code else ("README.md",)),
        snapshot_path=snapshot_path,
    )


def mark_download(
    model: str,
    *,
    kind: str,
    trust_remote_code: bool,
    allow_patterns: tuple[str, ...],
    snapshot_path: Path | None = None,
) -> None:
    marker_path = download_marker_path(model)
    if marker_path is None:
        return
    previous = read_download_marker(model)
    if previous and previous.get("kind") == "checkpoint" and kind == "analysis":
        return
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema": HF_MARKER_SCHEMA,
        "model": model,
        "kind": kind,
        "trust_remote_code": trust_remote_code,
        "allow_patterns": list(allow_patterns),
    }
    if snapshot_path is not None:
        marker["snapshot_path"] = str(snapshot_path)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cached_checkpoint_status(model: str) -> CachedCheckpointStatus:
    path = Path(model)
    if path.exists():
        return CachedCheckpointStatus(
            complete=True,
            kind="local",
            marker_path=None,
            snapshot_path=path,
        )
    marker_path = download_marker_path(model)
    marker = read_download_marker(model)
    if marker is None:
        return CachedCheckpointStatus(
            complete=False,
            kind="unknown",
            marker_path=marker_path,
            snapshot_path=None,
        )
    snapshot_value = marker.get("snapshot_path")
    snapshot_path = Path(snapshot_value) if isinstance(snapshot_value, str) else None
    return CachedCheckpointStatus(
        complete=marker.get("kind") == "checkpoint",
        kind=str(marker.get("kind") or "unknown"),
        marker_path=marker_path,
        snapshot_path=snapshot_path,
    )
