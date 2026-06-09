from __future__ import annotations

import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenpruner.architecture import ModelArchitectureReport
from tokenpruner.token_lists import compact_token_maps


class PruneRefusal(RuntimeError):
    pass


@dataclass(frozen=True)
class PruneResult:
    output_dir: Path
    retained_count: int
    pruned_count: int
    old_to_new: dict[int, int]
    new_to_old: dict[int, int]
    transformed_surfaces: tuple["TransformedSurface", ...] = ()


@dataclass(frozen=True)
class TransformedSurface:
    name: str
    kind: str
    shared_with: str | None
    tensors: tuple[str, ...]
    quantized: bool


def validate_pruning_architecture(report: ModelArchitectureReport, tokenizer_vocab_size: int) -> None:
    declared_sizes = {size for _, size in report.vocabulary_surfaces}
    too_small = [(path, size) for path, size in report.vocabulary_surfaces if size < tokenizer_vocab_size]
    if too_small:
        details = ", ".join(f"{path}={size}" for path, size in too_small)
        raise PruneRefusal(
            "Refusing unresolved vocabulary surface(s) smaller than the tokenizer vocabulary: "
            f"{details}. A separate token mapping is required."
        )
    if len(declared_sizes) > 1:
        details = ", ".join(f"{path}={size}" for path, size in report.vocabulary_surfaces)
        raise PruneRefusal(
            "Refusing multiple differently sized vocabulary surfaces: "
            f"{details}. A separate token mapping is required."
        )


def model_vocabulary_size(report: ModelArchitectureReport, tokenizer_vocab_size: int) -> int:
    validate_pruning_architecture(report, tokenizer_vocab_size)
    if not report.vocabulary_surfaces:
        return tokenizer_vocab_size
    return report.vocabulary_surfaces[0][1]


def validate_pruning_runtime(report: ModelArchitectureReport) -> None:
    missing = missing_pruning_packages(report)
    if missing:
        raise PruneRefusal("Missing pruning runtime package(s): " + ", ".join(missing))


def missing_pruning_packages(report: ModelArchitectureReport) -> tuple[str, ...]:
    missing: list[str] = []
    if not _module_available("torch"):
        missing.append("torch")
    method = (report.quantization_method or "").lower()
    if method == "modelopt" and not _module_available("modelopt"):
        missing.append("nvidia-modelopt")
    if method in {"compressed-tensors", "compressed_tensors"} and not _module_available("compressed_tensors"):
        missing.append("compressed-tensors")
    return tuple(missing)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def validate_text_only_pruning(
    report: ModelArchitectureReport,
    tokenizer_vocab_size: int | None = None,
) -> None:
    validate_pruning_architecture(
        report,
        tokenizer_vocab_size if tokenizer_vocab_size is not None else _primary_vocab_size(report),
    )


def _primary_vocab_size(report: ModelArchitectureReport) -> int:
    if not report.vocabulary_surfaces:
        raise PruneRefusal("Model configuration does not declare a vocabulary size.")
    return report.vocabulary_surfaces[0][1]


def load_causal_lm(
    model: str,
    trust_remote_code: bool = False,
    architecture_report: ModelArchitectureReport | None = None,
    **kwargs: Any,
) -> Any:
    try:
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and transformers model classes are required to prune model weights. "
            "Install a compatible torch build and transformers first."
        ) from exc

    transformers_logging.set_verbosity_error()
    architectures = architecture_report.architectures if architecture_report is not None else ()
    try:
        if any("ConditionalGeneration" in architecture for architecture in architectures):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(model, trust_remote_code=trust_remote_code, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "modelopt" in message.lower() or "nvfp4" in message.lower():
            raise RuntimeError(
                "This quantized checkpoint requires NVIDIA ModelOpt support. "
                "Install a compatible PyTorch build and nvidia-modelopt, then retry."
            ) from exc
        if "compressed_tensors" in message.lower() or "compressed-tensors" in message.lower():
            raise RuntimeError(
                "This quantized checkpoint requires compressed-tensors support. "
                "Install a compatible PyTorch build and compressed-tensors, then retry."
            ) from exc
        raise


def prune_model_and_tokenizer(
    *,
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    retained_token_ids: list[int],
    original_vocab_size: int,
    model_vocab_size: int | None = None,
) -> PruneResult:
    old_to_new, new_to_old = compact_token_maps(retained_token_ids)
    if not old_to_new:
        raise PruneRefusal("Refusing to prune to an empty vocabulary.")
    if max(old_to_new) >= original_vocab_size:
        raise PruneRefusal("Retained token IDs exceed the tokenizer vocabulary size.")
    model_vocab_size = model_vocab_size or original_vocab_size

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transformed_surfaces = slice_model_vocab(
        model,
        sorted(old_to_new),
        old_to_new,
        original_vocab_size=model_vocab_size,
    )
    remap_model_config(model.config, old_to_new, model_vocab_size)
    save_remapped_tokenizer(tokenizer, output, old_to_new)
    model.save_pretrained(output)
    write_pruning_maps(
        output,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        original_vocab_size=original_vocab_size,
        transformed_surfaces=transformed_surfaces,
    )
    return PruneResult(
        output_dir=output,
        retained_count=len(old_to_new),
        pruned_count=original_vocab_size - len(old_to_new),
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        transformed_surfaces=transformed_surfaces,
    )


def slice_model_vocab(
    model: Any,
    retained_token_ids: list[int],
    old_to_new: dict[int, int],
    original_vocab_size: int | None = None,
) -> tuple[TransformedSurface, ...]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to prune model weights.") from exc

    original_vocab_size = original_vocab_size or _model_vocab_size(model) or (max(old_to_new) + 1)
    surfaces = discover_vocabulary_surfaces(model, original_vocab_size)
    if not surfaces:
        raise PruneRefusal("No vocabulary-bearing embedding or output-head modules were found.")

    index_cache: dict[Any, Any] = {}
    sliced_tensors: dict[int, Any] = {}
    tensor_owner: dict[int, str] = {}
    transformed: list[TransformedSurface] = []

    for name, kind, module in surfaces:
        tensor_names: list[str] = []
        shared_with: str | None = None
        for tensor_name, tensor, is_parameter in _module_row_tensors(module, original_vocab_size):
            tensor_id = id(tensor)
            if tensor_id in sliced_tensors:
                replacement = sliced_tensors[tensor_id]
                shared_with = shared_with or tensor_owner[tensor_id]
            else:
                index = index_cache.setdefault(
                    tensor.device,
                    torch.tensor(retained_token_ids, dtype=torch.long, device=tensor.device),
                )
                sliced = tensor.detach().index_select(0, index).contiguous()
                replacement = torch.nn.Parameter(sliced, requires_grad=tensor.requires_grad) if is_parameter else sliced
                sliced_tensors[tensor_id] = replacement
                tensor_owner[tensor_id] = name
            _set_direct_tensor(module, tensor_name, replacement, is_parameter)
            tensor_names.append(tensor_name)

        if not tensor_names:
            raise PruneRefusal(
                f"Vocabulary surface '{name}' ({type(module).__name__}) exposes no row tensor "
                f"with first dimension {original_vocab_size}."
            )
        _update_module_vocab_metadata(module, len(retained_token_ids), old_to_new)
        transformed.append(
            TransformedSurface(
                name=name,
                kind=kind,
                shared_with=shared_with,
                tensors=tuple(tensor_names),
                quantized=not isinstance(module, (torch.nn.Embedding, torch.nn.Linear)),
            )
        )

    return tuple(transformed)


def _model_vocab_size(model: Any) -> int | None:
    config = getattr(model, "config", None)
    for candidate in (config, getattr(config, "text_config", None)):
        value = getattr(candidate, "vocab_size", None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def discover_vocabulary_surfaces(model: Any, original_vocab_size: int) -> list[tuple[str, str, Any]]:
    candidates: list[tuple[str, str, Any]] = []
    if hasattr(model, "get_input_embeddings"):
        module = model.get_input_embeddings()
        if module is not None and _module_has_vocab_rows(module, original_vocab_size):
            candidates.append(("input_embeddings", "embedding", module))
    if hasattr(model, "get_output_embeddings"):
        module = model.get_output_embeddings()
        if module is not None and _module_has_vocab_rows(module, original_vocab_size):
            candidates.append(("output_embeddings", "output_head", module))

    if hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            if not name or not _looks_like_vocabulary_module(name):
                continue
            if _module_has_vocab_rows(module, original_vocab_size):
                candidates.append((name, _surface_kind(name), module))

    unique: list[tuple[str, str, Any]] = []
    seen: set[int] = set()
    for name, kind, module in candidates:
        if id(module) in seen:
            continue
        seen.add(id(module))
        unique.append((name, kind, module))
    return unique


def _looks_like_vocabulary_module(name: str) -> bool:
    lowered = name.lower()
    markers = ("embed_tokens", "word_embeddings", "token_embedding", "lm_head", "output_head")
    return any(marker in lowered for marker in markers)


def _surface_kind(name: str) -> str:
    lowered = name.lower()
    prefix = "mtp_" if any(marker in lowered for marker in ("mtp", "nextn", "multi_token")) else ""
    return prefix + ("embedding" if "embed" in lowered else "output_head")


def _module_has_vocab_rows(module: Any, original_vocab_size: int) -> bool:
    return any(True for _ in _module_row_tensors(module, original_vocab_size))


def _module_row_tensors(module: Any, original_vocab_size: int):
    if hasattr(module, "named_parameters"):
        for name, tensor in module.named_parameters(recurse=False):
            if getattr(tensor, "ndim", 0) >= 1 and tensor.shape[0] == original_vocab_size:
                yield name, tensor, True
    if hasattr(module, "named_buffers"):
        for name, tensor in module.named_buffers(recurse=False):
            if getattr(tensor, "ndim", 0) >= 1 and tensor.shape[0] == original_vocab_size:
                yield name, tensor, False


def _set_direct_tensor(module: Any, name: str, tensor: Any, is_parameter: bool) -> None:
    if is_parameter:
        module._parameters[name] = tensor
    else:
        module._buffers[name] = tensor


def _update_module_vocab_metadata(module: Any, new_vocab_size: int, old_to_new: dict[int, int]) -> None:
    for attribute in ("num_embeddings", "out_features", "vocab_size"):
        if hasattr(module, attribute):
            setattr(module, attribute, new_vocab_size)
    if hasattr(module, "padding_idx"):
        module.padding_idx = _remapped_padding_idx(module, old_to_new)


def save_remapped_tokenizer(tokenizer: Any, output_dir: str | Path, old_to_new: dict[int, int]) -> Path:
    output = Path(output_dir)
    tokenizer.save_pretrained(output)
    tokenizer_json = output / "tokenizer.json"
    if not tokenizer_json.exists():
        raise PruneRefusal("Tokenizer did not save tokenizer.json; v1 can only rewrite JSON tokenizers.")
    retained_tokens = rewrite_tokenizer_json(tokenizer_json, old_to_new)
    _rewrite_vocab_json(output / "vocab.json", old_to_new)
    _rewrite_added_tokens_json(output / "added_tokens.json", old_to_new)
    _rewrite_tokenizer_config(output / "tokenizer_config.json", old_to_new)
    _rewrite_merges_file(output / "merges.txt", retained_tokens)
    validate_saved_tokenizer(output, len(old_to_new))
    return tokenizer_json


def rewrite_tokenizer_json(path: str | Path, old_to_new: dict[int, int]) -> set[str]:
    tokenizer_path = Path(path)
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model")
    if not isinstance(model, dict):
        raise PruneRefusal("Unsupported tokenizer.json: missing model object.")

    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        model["vocab"] = {
            token: old_to_new[int(token_id)]
            for token, token_id in sorted(vocab.items(), key=lambda item: int(item[1]))
            if int(token_id) in old_to_new
        }
    elif isinstance(vocab, list):
        model["vocab"] = [
            entry
            for old_id, entry in enumerate(vocab)
            if old_id in old_to_new
        ]
    else:
        raise PruneRefusal("Unsupported tokenizer.json: model.vocab must be an object or list.")

    if not model["vocab"]:
        raise PruneRefusal("Tokenizer rewrite produced an empty vocabulary.")

    retained_tokens = (
        set(model["vocab"])
        if isinstance(model["vocab"], dict)
        else {str(entry[0]) for entry in model["vocab"] if isinstance(entry, list) and entry}
    )
    if isinstance(model.get("merges"), list):
        model["merges"] = [
            merge
            for merge in model["merges"]
            if _merge_is_retained(merge, retained_tokens)
        ]
    data["added_tokens"] = _remap_added_tokens(data.get("added_tokens", []), old_to_new)
    _remap_token_references(data, old_to_new)
    tokenizer_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return retained_tokens


def validate_saved_tokenizer(output_dir: str | Path, expected_vocab_size: int) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True, use_fast=True)
    except Exception as exc:
        raise PruneRefusal(f"Pruned tokenizer could not be reloaded: {exc}") from exc
    ids = sorted(int(token_id) for token_id in tokenizer.get_vocab().values())
    expected = list(range(expected_vocab_size))
    if ids != expected:
        raise PruneRefusal(
            "Pruned tokenizer IDs are not compact and contiguous after reload "
            f"(expected 0..{expected_vocab_size - 1})."
        )


def _rewrite_vocab_json(path: Path, old_to_new: dict[int, int]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PruneRefusal("Unsupported vocab.json: expected a token-to-ID object.")
    rewritten = {
        token: old_to_new[int(token_id)]
        for token, token_id in sorted(data.items(), key=lambda item: int(item[1]))
        if int(token_id) in old_to_new
    }
    path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rewrite_added_tokens_json(path: Path, old_to_new: dict[int, int]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return
    rewritten = {
        token: old_to_new[int(token_id)]
        for token, token_id in data.items()
        if isinstance(token_id, int) and token_id in old_to_new
    }
    path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rewrite_tokenizer_config(path: Path, old_to_new: dict[int, int]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    decoder = data.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        remapped: dict[str, Any] = {}
        for raw_id, entry in decoder.items():
            old_id = int(raw_id)
            if old_id not in old_to_new:
                continue
            updated = dict(entry) if isinstance(entry, dict) else entry
            if isinstance(updated, dict) and isinstance(updated.get("id"), int):
                updated["id"] = old_to_new[old_id]
            remapped[str(old_to_new[old_id])] = updated
        data["added_tokens_decoder"] = remapped
    _remap_json_token_id_fields(data, old_to_new)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _remap_json_token_id_fields(value: Any, old_to_new: dict[int, int]) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if _is_token_id_key(str(key)):
                value[key] = _remap_config_token_id(child, old_to_new, f"tokenizer_config.{key}")
            elif key != "added_tokens_decoder":
                _remap_json_token_id_fields(child, old_to_new)
    elif isinstance(value, list):
        for child in value:
            _remap_json_token_id_fields(child, old_to_new)


def _rewrite_merges_file(path: Path, retained_tokens: set[str]) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [line for line in lines if line.startswith("#")]
    merges = [line for line in lines if line and not line.startswith("#") and _merge_is_retained(line, retained_tokens)]
    path.write_text("\n".join(header + merges) + "\n", encoding="utf-8")


def _merge_is_retained(merge: Any, retained_tokens: set[str]) -> bool:
    if isinstance(merge, str):
        parts = merge.split()
    elif isinstance(merge, list):
        parts = [str(part) for part in merge]
    else:
        return False
    if len(parts) != 2:
        return False
    left, right = parts
    return left in retained_tokens and right in retained_tokens and left + right in retained_tokens


def write_pruning_maps(
    output_dir: str | Path,
    *,
    old_to_new: dict[int, int],
    new_to_old: dict[int, int],
    original_vocab_size: int,
    transformed_surfaces: tuple[TransformedSurface, ...] = (),
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
                "retained_vocab_size": len(old_to_new),
                "pruned_vocab_size": original_vocab_size - len(old_to_new),
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


def remap_model_config(config: Any, old_to_new: dict[int, int], original_vocab_size: int) -> None:
    _remap_config_value(config, old_to_new, original_vocab_size, path="config", seen=set())


def _remap_config_value(
    value: Any,
    old_to_new: dict[int, int],
    original_vocab_size: int,
    *,
    path: str,
    seen: set[int],
) -> None:
    if isinstance(value, (dict, list)) or hasattr(value, "__dict__"):
        if id(value) in seen:
            return
        seen.add(id(value))
    if isinstance(value, dict):
        items = list(value.items())
        for key, child in items:
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_token_id_key(key_text):
                value[key] = _remap_config_token_id(child, old_to_new, child_path)
            elif key_text == "vocab_size" and child == original_vocab_size:
                value[key] = len(old_to_new)
            else:
                _remap_config_value(child, old_to_new, original_vocab_size, path=child_path, seen=seen)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _remap_config_value(
                child,
                old_to_new,
                original_vocab_size,
                path=f"{path}[{index}]",
                seen=seen,
            )
        return
    if not hasattr(value, "__dict__"):
        return
    for key, child in list(vars(value).items()):
        child_path = f"{path}.{key}"
        if _is_token_id_key(key):
            setattr(value, key, _remap_config_token_id(child, old_to_new, child_path))
        elif key == "vocab_size" and child == original_vocab_size:
            setattr(value, key, len(old_to_new))
        elif not key.startswith("_"):
            _remap_config_value(child, old_to_new, original_vocab_size, path=child_path, seen=seen)


def _is_token_id_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_token_id") or lowered.endswith("_token_ids")


def _remap_config_token_id(value: Any, old_to_new: dict[int, int], path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, int):
        if value not in old_to_new:
            raise PruneRefusal(f"Required configuration token {path}={value} was not retained.")
        return old_to_new[value]
    if isinstance(value, (list, tuple)):
        remapped = [_remap_config_token_id(item, old_to_new, path) for item in value]
        return type(value)(remapped)
    return value


def _remapped_padding_idx(input_embeddings: Any, old_to_new: dict[int, int]) -> int | None:
    padding_idx = getattr(input_embeddings, "padding_idx", None)
    if padding_idx is None:
        return None
    return old_to_new.get(int(padding_idx))


def _remap_added_tokens(value: Any, old_to_new: dict[int, int]) -> list[Any]:
    if not isinstance(value, list):
        return []
    remapped = []
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), int):
            continue
        old_id = int(entry["id"])
        if old_id not in old_to_new:
            continue
        updated = dict(entry)
        updated["id"] = old_to_new[old_id]
        remapped.append(updated)
    return remapped


def _remap_token_references(value: Any, old_to_new: dict[int, int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"vocab", "added_tokens"}:
                continue
            if key == "id" and isinstance(child, int) and child in old_to_new:
                value[key] = old_to_new[child]
            else:
                _remap_token_references(child, old_to_new)
        return
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int) and value[1] in old_to_new:
            value[1] = old_to_new[value[1]]
            return
        for child in value:
            _remap_token_references(child, old_to_new)
