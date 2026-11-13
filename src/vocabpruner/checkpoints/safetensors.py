from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from vocabpruner.checkpoints.architecture import vocabulary_surface_kind, is_checkpoint_vocab_tensor
from vocabpruner.checkpoints.codebooks import codebook_config_ids, codebook_tensor_axis
from vocabpruner.checkpoints.config import remap_generation_config, write_pruned_config
from vocabpruner.errors import PruneRefusal, VocabRewriteError
from vocabpruner.mapping import compact_token_maps
from vocabpruner.models import PruneResult, TransformedSurface
from vocabpruner.storage.hub import resolve_checkpoint_source
from vocabpruner.storage.metadata import (
    checkpoint_file_size,
    copy_checkpoint_sidecars,
    write_pruning_maps,
)
from vocabpruner.tokenization.preservation import complete_retained_token_ids
from vocabpruner.tokenization.rewriting import save_remapped_tokenizer


def prune_safetensors_checkpoint(
    *,
    model_source: str,
    tokenizer: Any | None,
    config: Any,
    output_dir: str | Path,
    retained_token_ids: list[int],
    original_vocab_size: int,
    model_vocab_size: int | None = None,
    trust_remote_code: bool = False,
) -> PruneResult | None:
    source = resolve_checkpoint_source(model_source, trust_remote_code=trust_remote_code)
    index = load_safetensors_index(source)
    safetensor_paths = checkpoint_safetensor_paths(source, index)
    if not safetensor_paths:
        return None

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors and torch are required for streaming checkpoint pruning.") from exc

    validate_safetensor_files(safetensor_paths, safe_open)

    retained_token_ids = complete_retained_token_ids(tokenizer, retained_token_ids)
    old_to_new, new_to_old = compact_token_maps(retained_token_ids)
    if not old_to_new:
        raise PruneRefusal("Refusing to prune to an empty vocabulary.")
    if max(old_to_new) >= original_vocab_size:
        raise PruneRefusal("Retained token IDs exceed the tokenizer vocabulary size.")
    model_vocab_size = model_vocab_size or original_vocab_size

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_checkpoint_bytes = sum(path.stat().st_size for path in safetensor_paths)
    copy_checkpoint_sidecars(source, output)
    write_pruned_config(config, output / "config.json", old_to_new, model_vocab_size)
    remap_generation_config(output / "generation_config.json", old_to_new, model_vocab_size)
    if tokenizer is not None:
        try:
            save_remapped_tokenizer(tokenizer, output, old_to_new)
        except VocabRewriteError as exc:
            raise PruneRefusal(str(exc)) from exc

    total_size = 0
    grouped_surfaces: dict[str, dict[str, Any]] = {}
    retained = torch.tensor(sorted(old_to_new), dtype=torch.long)

    for source_path in safetensor_paths:
        tensors: dict[str, Any] = {}
        metadata = None
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                axis = checkpoint_vocab_axis(name, tensor.shape, model_vocab_size)
                if axis is not None:
                    tensor = tensor.index_select(axis, retained).contiguous()
                    record_checkpoint_surface(grouped_surfaces, name)
                tensors[name] = tensor
                total_size += int(tensor.numel() * tensor.element_size())
        save_file(tensors, output / source_path.name, metadata=metadata)

    if not grouped_surfaces and not codebook_config_ids(config):
        raise PruneRefusal("No vocabulary-bearing safetensors were found to slice.")
    write_safetensors_index(output, index, total_size)
    pruned_checkpoint_bytes = checkpoint_file_size(output)
    transformed_surfaces = tuple(
        TransformedSurface(
            name=name,
            kind=str(info["kind"]),
            shared_with=None,
            tensors=tuple(sorted(info["tensors"])),
            quantized=bool(info["quantized"]),
        )
        for name, info in sorted(grouped_surfaces.items())
    )
    write_pruning_maps(
        output,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        original_vocab_size=original_vocab_size,
        source_model_vocab_size=model_vocab_size,
        transformed_surfaces=transformed_surfaces,
        original_checkpoint_bytes=original_checkpoint_bytes,
        pruned_checkpoint_bytes=pruned_checkpoint_bytes,
    )
    return PruneResult(
        output_dir=output,
        retained_count=len(old_to_new),
        pruned_count=original_vocab_size - len(old_to_new),
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        transformed_surfaces=transformed_surfaces,
        original_checkpoint_bytes=original_checkpoint_bytes,
        pruned_checkpoint_bytes=pruned_checkpoint_bytes,
    )


def load_safetensors_index(source: Path) -> dict[str, Any] | None:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


def checkpoint_safetensor_paths(source: Path, index: dict[str, Any] | None) -> list[Path]:
    if index is None:
        return sorted(source.glob("*.safetensors"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PruneRefusal("Unsupported safetensors index: missing weight_map object.")
    filenames = sorted({str(filename) for filename in weight_map.values()})
    paths = [source / filename for filename in filenames]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise PruneRefusal(
            "Checkpoint snapshot is incomplete; safetensors index references missing file(s): "
            + ", ".join(missing)
        )
    empty = [path.name for path in paths if path.stat().st_size == 0]
    if empty:
        raise PruneRefusal(
            "Checkpoint snapshot is incomplete; safetensors file(s) are empty: "
            + ", ".join(empty)
        )
    return paths


def validate_safetensor_files(paths: list[Path], safe_open: Any) -> None:
    invalid: list[str] = []
    for path in paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                tuple(handle.keys())
        except Exception as exc:
            invalid.append(f"{path.name} ({exc})")
    if invalid:
        raise PruneRefusal(
            "Checkpoint snapshot is incomplete or corrupt; safetensors file(s) could not be opened: "
            + "; ".join(invalid)
        )


def write_safetensors_index(output: Path, index: dict[str, Any] | None, total_size: int) -> None:
    if index is None:
        return
    updated = dict(index)
    metadata = dict(updated.get("metadata") or {})
    metadata["total_size"] = total_size
    updated["metadata"] = metadata
    (output / "model.safetensors.index.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def checkpoint_vocab_axis(name: str, shape: Any, original_vocab_size: int) -> int | None:
    if is_checkpoint_vocab_tensor(name, shape, original_vocab_size):
        return 0
    return codebook_tensor_axis(name, shape, original_vocab_size)


def record_checkpoint_surface(grouped: dict[str, dict[str, Any]], tensor_name: str) -> None:
    module_name = checkpoint_module_name(tensor_name)
    entry = grouped.setdefault(
        module_name,
        {
            "kind": vocabulary_surface_kind(module_name),
            "tensors": [],
            "quantized": False,
        },
    )
    entry["tensors"].append(tensor_name.rsplit(".", 1)[-1])
    if not tensor_name.endswith(".weight") and not tensor_name.endswith(".bias"):
        entry["quantized"] = True


def checkpoint_module_name(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    if len(parts) <= 1:
        return tensor_name
    if parts[-1] in {"weight", "bias"}:
        return ".".join(parts[:-1])
    if len(parts) >= 2 and parts[-2] in {"weight", "bias"}:
        return ".".join(parts[:-2])
    return ".".join(parts[:-1])
