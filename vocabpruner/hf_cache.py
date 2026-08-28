from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vocabpruner.paths import hf_models_dir

HF_DOWNLOAD_MARKER = "hf_checkpoint_manifest.json"
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
CHECKPOINT_ALLOW_PATTERNS = ("*",)
CHECKPOINT_IGNORE_PATTERNS = (
    "*.bin",
    "*.ckpt",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.pb",
    "*.pt",
    "*.pth",
    "*.tflite",
)
SHARD_NAME = re.compile(r"^(.*)-(\d+)-of-(\d+)\.(safetensors|bin)$")


@dataclass(frozen=True)
class CachedCheckpointStatus:
    complete: bool
    kind: str
    marker_path: Path | None
    snapshot_path: Path | None


def repo_cache_dir(model: str) -> Path | None:
    if Path(model).exists():
        return None
    return hf_models_dir() / f"models--{model.replace('/', '--')}"


def download_marker_path(model: str) -> Path | None:
    cache_dir = repo_cache_dir(model)
    return None if cache_dir is None else cache_dir / HF_DOWNLOAD_MARKER


def checkpoint_inventory(root: Path) -> list[dict[str, int | str]]:
    root = root.resolve()
    files: set[Path] = set()
    for index_path in (root / "model.safetensors.index.json", root / "pytorch_model.bin.index.json"):
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid checkpoint index: {index_path.name}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"empty checkpoint index: {index_path.name}")
        files.update(root / str(name) for name in weight_map.values())
    files.update(root.glob("*.safetensors"))
    files.update(root.glob("*.gguf"))
    files.update(root.glob("pytorch_model*.bin"))
    if not files:
        raise ValueError("no checkpoint weights found")
    for path in tuple(files):
        match = SHARD_NAME.match(path.name)
        if match:
            prefix, _, count, suffix = match.groups()
            files.update(
                root / f"{prefix}-{number:05d}-of-{int(count):05d}.{suffix}"
                for number in range(1, int(count) + 1)
            )
    if any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError("checkpoint has missing or empty weights")
    return [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size}
        for path in sorted(files)
    ]


def write_checkpoint_manifest(root: str | Path, *, model: str, snapshot_path: str | Path) -> Path:
    root = Path(root).resolve()
    snapshot_path = Path(snapshot_path).resolve()
    marker_path = root / HF_DOWNLOAD_MARKER
    payload: dict[str, Any] = {
        "schema": HF_MARKER_SCHEMA,
        "kind": "checkpoint",
        "model": model,
        "snapshot_path": str(snapshot_path),
        "files": checkpoint_inventory(snapshot_path),
    }
    temporary = marker_path.with_name(f".{marker_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker_path)
    return marker_path


def read_download_marker(model: str) -> dict[str, Any] | None:
    marker_path = download_marker_path(model)
    if marker_path is None or not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) and marker.get("schema") == HF_MARKER_SCHEMA else None


def mark_analysis_download(model: str, trust_remote_code: bool) -> None:
    del model, trust_remote_code


def mark_checkpoint_download(model: str, trust_remote_code: bool, snapshot_path: Path) -> None:
    del trust_remote_code
    cache_dir = repo_cache_dir(model)
    if cache_dir is not None:
        write_checkpoint_manifest(cache_dir, model=model, snapshot_path=snapshot_path)


def _valid_snapshot(path: Path, expected: Any = None) -> bool:
    try:
        actual = checkpoint_inventory(path)
    except (OSError, ValueError):
        return False
    return expected is None or actual == expected


def _snapshot_candidates(cache_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    ref = cache_dir / "refs" / "main"
    if ref.is_file():
        try:
            candidates.append(cache_dir / "snapshots" / ref.read_text(encoding="utf-8").strip())
        except OSError:
            pass
    snapshots = cache_dir / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted((path for path in snapshots.iterdir() if path.is_dir()), reverse=True))
    return list(dict.fromkeys(candidates))


def cached_checkpoint_status(model: str) -> CachedCheckpointStatus:
    path = Path(model)
    if path.exists():
        return CachedCheckpointStatus(
            complete=_valid_snapshot(path),
            kind="local",
            marker_path=None,
            snapshot_path=path,
        )
    cache_dir = repo_cache_dir(model)
    marker_path = download_marker_path(model)
    marker = read_download_marker(model)
    if marker is not None:
        snapshot_value = marker.get("snapshot_path")
        snapshot = Path(snapshot_value) if isinstance(snapshot_value, str) else None
        if snapshot is not None and _valid_snapshot(snapshot, marker.get("files")):
            return CachedCheckpointStatus(True, "checkpoint", marker_path, snapshot)
    if cache_dir is not None:
        for snapshot in _snapshot_candidates(cache_dir):
            if _valid_snapshot(snapshot):
                return CachedCheckpointStatus(True, "checkpoint", marker_path, snapshot)
    return CachedCheckpointStatus(False, "unknown", marker_path, None)
