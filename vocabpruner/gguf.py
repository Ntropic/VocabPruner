from __future__ import annotations

from pathlib import Path
from typing import Any

from vocabpruner.paths import hf_models_dir


def is_gguf(model: str | Path) -> bool:
    return resolve_gguf(model) is not None


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


def _reader(model: str | Path):
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("GGUF support requires the gguf package. Install project dependencies first.") from exc
    return GGUFReader(model)


def _field(reader: Any, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    return default if field is None else field.contents()


class GGUFTokenizer:
    def __init__(self, model: str | Path):
        reader = _reader(model)
        self.tokens = [str(token) for token in _field(reader, "tokenizer.ggml.tokens", [])]
        if not self.tokens:
            raise ValueError("GGUF has no tokenizer.ggml.tokens metadata.")
        token_types = _field(reader, "tokenizer.ggml.token_type", [1] * len(self.tokens))
        self.all_special_ids = [index for index, kind in enumerate(token_types) if int(kind) in {2, 3, 4}]
        self.all_special_tokens = [self.tokens[index] for index in self.all_special_ids]
        self.special_tokens_map = {}
        self.model = str(_field(reader, "tokenizer.ggml.model", ""))

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        tokens = [self.tokens[int(token_id)] for token_id in token_ids]
        if self.model == "gpt2":
            from tokenizers.decoders import ByteLevel

            return ByteLevel().decode(tokens)
        text = "".join(tokens).replace("▁", " ")
        if len(tokens) == 1 and tokens[0].startswith("<0x") and tokens[0].endswith(">"):
            try:
                return bytes([int(tokens[0][3:-1], 16)]).decode("latin-1")
            except ValueError:
                pass
        return text


def config_dict(model: str | Path) -> dict[str, Any]:
    reader = _reader(model)
    architecture = str(_field(reader, "general.architecture", "unknown"))
    tokens = _field(reader, "tokenizer.ggml.tokens", [])
    return {
        "model_type": architecture,
        "architectures": [],
        "vocab_size": len(tokens),
        "quantization_config": {"quant_method": "gguf"},
    }


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

    reader = _reader(source)
    tokens = list(_field(reader, "tokenizer.ggml.tokens", []))
    old_to_new = {old: new for new, old in enumerate(retained_ids)}
    if not tokens or not old_to_new or max(old_to_new) >= len(tokens):
        raise ValueError("Invalid retained token IDs for GGUF tokenizer.")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture = str(_field(reader, "general.architecture", "unknown"))
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
    return output, tuple(transformed)


def _merge_retained(merge: str, tokens: set[str]) -> bool:
    parts = merge.split()
    return len(parts) == 2 and parts[0] in tokens and parts[1] in tokens and "".join(parts) in tokens
