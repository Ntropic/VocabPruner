from __future__ import annotations

from pathlib import Path

from vocabpruner.errors import PruneRefusal
from vocabpruner.mapping import compact_token_maps
from vocabpruner.models import PruneResult, TransformedSurface
from vocabpruner.storage.gguf import gguf_field, read_gguf, resolve_gguf
from vocabpruner.storage.metadata import write_pruning_maps
from vocabpruner.tokenization.gguf import GGUFTokenizer
from vocabpruner.tokenization.preservation import complete_retained_token_ids
from vocabpruner.tokenization.validation import validate_gguf_tokenizer


def prune_gguf_checkpoint(
    *,
    model_source: str,
    output_dir: str | Path,
    retained_token_ids: list[int],
    original_vocab_size: int,
) -> PruneResult | None:
    source = resolve_gguf(model_source)
    if source is None:
        return None
    retained_token_ids = complete_retained_token_ids(GGUFTokenizer(source), retained_token_ids)
    old_to_new, new_to_old = compact_token_maps(retained_token_ids)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{source.stem}-pruned.gguf"
    try:
        rewritten, tensor_names = rewrite(source, destination, sorted(old_to_new))
    except ValueError as exc:
        raise PruneRefusal(str(exc)) from exc
    surfaces = tuple(
        TransformedSurface(
            name=name,
            kind="embedding" if name == "token_embd.weight" else "output",
            shared_with=None,
            tensors=(name,),
            quantized=True,
        )
        for name in tensor_names
    )
    if not surfaces:
        raise PruneRefusal("No vocabulary-bearing GGUF tensors were found to slice.")
    write_pruning_maps(
        output,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        original_vocab_size=original_vocab_size,
        source_model_vocab_size=original_vocab_size,
        transformed_surfaces=surfaces,
        original_checkpoint_bytes=source.stat().st_size,
        pruned_checkpoint_bytes=rewritten.stat().st_size,
    )
    return PruneResult(
        output_dir=output,
        retained_count=len(old_to_new),
        pruned_count=original_vocab_size - len(old_to_new),
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        transformed_surfaces=surfaces,
        original_checkpoint_bytes=source.stat().st_size,
        pruned_checkpoint_bytes=rewritten.stat().st_size,
    )


def rewrite(
    source: str | Path,
    destination: str | Path,
    retained_ids: list[int],
) -> tuple[Path, tuple[str, ...]]:
    try:
        from gguf import GGUFValueType, GGUFWriter
    except ImportError as exc:
        raise RuntimeError("GGUF support requires the gguf package. Install project dependencies first.") from exc

    import numpy as np

    class Writer(GGUFWriter):
        # ponytail: gguf 0.19 rejects empty arrays; delete this override when its writer supports them.
        def _pack_val(self, val, vtype, add_vtype, sub_type=None):
            if vtype == GGUFValueType.ARRAY and not val and sub_type is not None:
                prefix = self._pack("I", vtype) if add_vtype else b""
                return prefix + self._pack("I", sub_type) + self._pack("Q", 0)
            return super()._pack_val(val, vtype, add_vtype, sub_type)

    reader = read_gguf(source)
    tokens = list(gguf_field(reader, "tokenizer.ggml.tokens", []))
    from vocabpruner.tokenization.preservation import complete_retained_token_ids

    if set(complete_retained_token_ids(GGUFTokenizer(source), retained_ids)) != set(retained_ids):
        raise ValueError("GGUF token map omits required byte tokens or BPE merge dependencies.")
    old_to_new = {old: new for new, old in enumerate(retained_ids)}
    if not tokens or not old_to_new or max(old_to_new) >= len(tokens):
        raise ValueError("Invalid retained token IDs for GGUF tokenizer.")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture = str(gguf_field(reader, "general.architecture", "unknown"))
    writer = Writer(output, architecture, endianess=reader.endianess)
    retained = np.asarray(retained_ids, dtype=np.int64)
    retained_tokens = {str(tokens[index]) for index in retained_ids}

    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        if key in {"tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type"}:
            value = [value[index] for index in retained_ids]
        elif key == "tokenizer.ggml.merges":
            value = [merge for merge in value if _merge_retained(str(merge), retained_tokens)]
        elif key.endswith(".vocab_size"):
            value = len(retained_ids)
        elif key.endswith("_token_id") and isinstance(value, int):
            if value not in old_to_new:
                continue
            value = old_to_new[value]
        subtype = field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None
        writer.add_key_value(key, value, field.types[0], sub_type=subtype)

    transformed: list[str] = []
    for tensor in reader.tensors:
        data = tensor.data
        if tensor.shape[-1] == len(tokens) and tensor.name in {"token_embd.weight", "output.weight"}:
            data = data[retained]
            transformed.append(tensor.name)
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()
    validate_gguf_tokenizer(source, output, retained_ids)
    return output, tuple(transformed)


def _merge_retained(merge: str, tokens: set[str]) -> bool:
    parts = merge.split()
    return len(parts) == 2 and parts[0] in tokens and parts[1] in tokens and "".join(parts) in tokens
