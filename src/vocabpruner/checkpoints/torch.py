from __future__ import annotations

from pathlib import Path
from typing import Any

from vocabpruner.checkpoints.architecture import (
    vocabulary_surface_kind,
    is_checkpoint_vocab_tensor,
    ModelArchitectureReport,
)
from vocabpruner.checkpoints.config import remap_generation_config, remap_model_config
from vocabpruner.errors import PruneRefusal, VocabRewriteError
from vocabpruner.mapping import compact_token_maps
from vocabpruner.models import PruneResult, TransformedSurface
from vocabpruner.storage.metadata import (
    checkpoint_file_size,
    copy_checkpoint_sidecars,
    write_pruning_maps,
)
from vocabpruner.storage.paths import hf_models_dir
from vocabpruner.tokenization.preservation import complete_retained_token_ids
from vocabpruner.tokenization.rewriting import save_remapped_tokenizer


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
                cache_dir=hf_models_dir(),
                **kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            cache_dir=hf_models_dir(),
            **kwargs,
        )
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
    source_dir: str | Path | None = None,
) -> PruneResult:
    retained_token_ids = complete_retained_token_ids(tokenizer, retained_token_ids)
    old_to_new, new_to_old = compact_token_maps(retained_token_ids)
    if not old_to_new:
        raise PruneRefusal("Refusing to prune to an empty vocabulary.")
    if max(old_to_new) >= original_vocab_size:
        raise PruneRefusal("Retained token IDs exceed the tokenizer vocabulary size.")
    model_vocab_size = model_vocab_size or original_vocab_size

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        copy_checkpoint_sidecars(Path(source_dir), output)
    try:
        save_remapped_tokenizer(tokenizer, output, old_to_new)
    except VocabRewriteError as exc:
        raise PruneRefusal(str(exc)) from exc
    transformed_surfaces = slice_model_vocab(
        model,
        sorted(old_to_new),
        old_to_new,
        original_vocab_size=model_vocab_size,
    )
    model.save_pretrained(output)
    remap_generation_config(output / "generation_config.json", old_to_new, model_vocab_size)
    pruned_checkpoint_bytes = checkpoint_file_size(output)
    write_pruning_maps(
        output,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        original_vocab_size=original_vocab_size,
        source_model_vocab_size=model_vocab_size,
        transformed_surfaces=transformed_surfaces,
        original_checkpoint_bytes=None,
        pruned_checkpoint_bytes=pruned_checkpoint_bytes,
    )
    return PruneResult(
        output_dir=output,
        retained_count=len(old_to_new),
        pruned_count=original_vocab_size - len(old_to_new),
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        transformed_surfaces=transformed_surfaces,
        pruned_checkpoint_bytes=pruned_checkpoint_bytes,
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

    config = getattr(model, "config", None)
    if config is not None:
        remap_model_config(config, old_to_new, original_vocab_size)
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
            if not name or not is_checkpoint_vocab_tensor(name):
                continue
            if _module_has_vocab_rows(module, original_vocab_size):
                candidates.append((name, vocabulary_surface_kind(name), module))

    unique: list[tuple[str, str, Any]] = []
    seen: set[int] = set()
    for name, kind, module in candidates:
        if id(module) in seen:
            continue
        seen.add(id(module))
        unique.append((name, kind, module))
    return unique


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


def _remapped_padding_idx(input_embeddings: Any, old_to_new: dict[int, int]) -> int | None:
    padding_idx = getattr(input_embeddings, "padding_idx", None)
    if padding_idx is None:
        return None
    return old_to_new.get(int(padding_idx))
