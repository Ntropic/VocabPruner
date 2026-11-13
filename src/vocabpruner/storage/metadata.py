from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json
import shutil

from vocabpruner.errors import PruneRefusal, VocabRewriteError
from vocabpruner.mapping import compact_token_maps
from vocabpruner.models import TransformedSurface
from vocabpruner.storage.hub import CHECKPOINT_IGNORE_PATTERNS
from vocabpruner.storage.paths import pruned_models_dir
from vocabpruner.tokenization.preservation import required_byte_token_ids


MODEL_METADATA_FILE = "vocabpruner_model.json"


def save_related_tokenizer(related_model: Path, output: Path, retained_count: int) -> None:
    from vocabpruner.tokenization.loading import load_tokenizer

    tokenizer = load_tokenizer(str(related_model))
    if set(tokenizer.get_vocab().values()) != set(range(retained_count)):
        raise PruneRefusal("Related tokenizer vocabulary does not match the inherited compact map.")
    tokenizer.save_pretrained(output)
    report = related_model / "tokenizer_validation.json"
    if report.is_file():
        validation = json.loads(report.read_text(encoding="utf-8"))
        validation["related_model"] = str(related_model.resolve())
        validation["scope"] = "Inherited tokenizer vocabulary validation; draft model inference is not validated"
        (output / "tokenizer_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def write_prune_metadata(
    pruned_model: str | Path,
    source_model: str,
    score_config: Any,
    related_model: Path | None = None,
) -> Path:
    path = Path(pruned_model) / MODEL_METADATA_FILE
    payload: dict[str, Any] = {
        "source_model": str(source_model),
        "pruned_model": str(Path(pruned_model).absolute()),
    }
    if related_model is not None:
        try:
            related_metadata = json.loads((related_model / MODEL_METADATA_FILE).read_text(encoding="utf-8"))
            inherited_config = related_metadata["score_config"]
            if not isinstance(inherited_config, dict) or not inherited_config:
                raise ValueError("missing selection configuration")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PruneRefusal(f"Cannot inherit related model selection metadata: {related_model}") from exc
        payload["score_config"] = inherited_config
        payload["related_model"] = str(related_model.resolve())
    elif is_dataclass(score_config):
        payload["score_config"] = _to_jsonable(asdict(score_config))
    else:
        payload["score_config"] = _to_jsonable(dict(score_config)) if isinstance(score_config, dict) else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_related_token_ids(directory: Path, source_vocab_size: int) -> list[int]:
    tokenizer_path = directory / "tokenizer.json"
    if tokenizer_path.is_file():
        try:
            required_byte_token_ids(json.loads(tokenizer_path.read_text(encoding="utf-8")))
        except VocabRewriteError as exc:
            raise PruneRefusal(f"Related model has an unsafe tokenizer; regenerate it: {exc}") from exc
    map_path = directory / "old_to_new_token_ids.json"
    report_path = directory / "pruning_report.json"
    if not map_path.is_file() or not report_path.is_file():
        raise PruneRefusal(f"Related model directory lacks pruning map metadata: {directory}")
    try:
        raw_map = json.loads(map_path.read_text(encoding="utf-8"))
        old_to_new = {int(old): int(new) for old, new in raw_map.items()}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reference_vocab_size = int(report["source_model_vocab_size"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PruneRefusal(f"Invalid related pruning map metadata; regenerate the related model: {directory}") from exc
    expected, _ = compact_token_maps(list(old_to_new))
    if not old_to_new or old_to_new != expected:
        raise PruneRefusal("Related token map must be a nonempty compact old-ID to new-ID map.")
    if reference_vocab_size != source_vocab_size:
        raise PruneRefusal(
            f"Related source vocabulary has {reference_vocab_size} tokens; model has {source_vocab_size}."
        )
    if min(old_to_new) < 0 or max(old_to_new) >= source_vocab_size:
        raise PruneRefusal("Related token map contains IDs outside the source vocabulary.")
    return sorted(old_to_new)


def resolve_related_model_dir(model_or_path: str) -> Path:
    path = Path(model_or_path).expanduser()
    if path.exists() or path.is_absolute() or model_or_path.startswith((".", "~")):
        return path.resolve()
    name = path.name
    return pruned_models_dir() / name


def copy_checkpoint_sidecars(source: Path, output: Path) -> None:
    skip_suffixes = {".safetensors", *(pattern[1:] for pattern in CHECKPOINT_IGNORE_PATTERNS)}
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for path in source.iterdir():
        if path.name in skip_names or (path.suffix in skip_suffixes and path.name != "modelopt_state.pth"):
            continue
        destination = output / path.name
        if path.is_dir():
            if destination.exists():
                continue
            shutil.copytree(
                path,
                destination,
                ignore=shutil.ignore_patterns("*.safetensors", *CHECKPOINT_IGNORE_PATTERNS),
            )
        elif path.is_file():
            shutil.copy2(path, destination)


def checkpoint_file_size(path: str | Path) -> int | None:
    root = Path(path)
    if not root.exists():
        return None
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    if root.is_file():
        return root.stat().st_size if root.suffix in weight_suffixes else None
    total = 0
    for child in root.rglob("*"):
        if child.is_file() and child.suffix in weight_suffixes:
            total += child.stat().st_size
    return total


def write_pruning_maps(
    output_dir: str | Path,
    *,
    old_to_new: dict[int, int],
    new_to_old: dict[int, int],
    original_vocab_size: int,
    source_model_vocab_size: int | None = None,
    transformed_surfaces: tuple[TransformedSurface, ...] = (),
    original_checkpoint_bytes: int | None = None,
    pruned_checkpoint_bytes: int | None = None,
) -> None:
    output = Path(output_dir)
    (output / "old_to_new_token_ids.json").write_text(
        json.dumps({str(old): new for old, new in sorted(old_to_new.items())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "new_to_old_token_ids.json").write_text(
        json.dumps({str(new): old for new, old in sorted(new_to_old.items())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "pruning_report.json").write_text(
        json.dumps(
            {
                "original_vocab_size": original_vocab_size,
                "source_model_vocab_size": source_model_vocab_size or original_vocab_size,
                "retained_vocab_size": len(old_to_new),
                "pruned_vocab_size": original_vocab_size - len(old_to_new),
                "original_checkpoint_bytes": original_checkpoint_bytes,
                "pruned_checkpoint_bytes": pruned_checkpoint_bytes,
                "checkpoint_size_reduction_bytes": (
                    original_checkpoint_bytes - pruned_checkpoint_bytes
                    if original_checkpoint_bytes is not None and pruned_checkpoint_bytes is not None
                    else None
                ),
                "id_strategy": "compact",
                "transformed_surfaces": [
                    {
                        "name": surface.name,
                        "kind": surface.kind,
                        "shared_with": surface.shared_with,
                        "tensors": list(surface.tensors),
                        "quantized": surface.quantized,
                    }
                    for surface in transformed_surfaces
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
