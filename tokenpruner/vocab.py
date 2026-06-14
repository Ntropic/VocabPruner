from __future__ import annotations

import json
import re
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tokenpruner.hf_cache import mark_analysis_download
from tokenpruner.models import TokenRecord
from tokenpruner.paths import HF_MODELS_DIR

BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


class VocabRewriteError(RuntimeError):
    pass


def is_byte_fallback_token(raw_token: str) -> bool:
    if BYTE_FALLBACK_RE.match(raw_token):
        return True
    if len(raw_token) == 1 and 0 <= ord(raw_token) <= 255 and raw_token.isprintable() is False:
        return True
    return False


def load_tokenizer(model: str, trust_remote_code: bool = False, **kwargs: Any) -> Any:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from transformers import AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load model tokenizers. Install project dependencies first."
        ) from exc

    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        cache_dir=HF_MODELS_DIR,
        **kwargs,
    )
    mark_analysis_download(model, trust_remote_code)
    return tokenizer


def tokenizer_records(tokenizer: Any) -> list[TokenRecord]:
    vocab = tokenizer.get_vocab()
    special_ids = _special_token_ids(tokenizer)
    records: list[TokenRecord] = []
    for raw_token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        decoded = decode_token(tokenizer, token_id, raw_token)
        records.append(
            TokenRecord(
                token_id=int(token_id),
                raw_token=str(raw_token),
                decoded_text=decoded,
                is_special=int(token_id) in special_ids or str(raw_token) in _special_tokens(tokenizer),
                is_byte_fallback=is_byte_fallback_token(str(raw_token)),
            )
        )
    return records


def decode_token(tokenizer: Any, token_id: int, raw_token: str) -> str:
    try:
        decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        decoded = ""
    if decoded:
        return decoded
    return raw_token


def vocabulary_stats(records: list[TokenRecord]) -> dict[str, int]:
    return {
        "vocab_size": len(records),
        "special_tokens": sum(1 for record in records if record.is_special),
        "byte_fallback_tokens": sum(1 for record in records if record.is_byte_fallback),
    }


def _special_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "all_special_ids", None) or []
    return {int(value) for value in values if value is not None}


def _special_tokens(tokenizer: Any) -> set[str]:
    tokens = set(getattr(tokenizer, "all_special_tokens", None) or [])
    added = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(added, Mapping):
        for value in added.values():
            if isinstance(value, str):
                tokens.add(value)
            elif isinstance(value, list):
                tokens.update(str(item) for item in value)
    return tokens


def save_remapped_tokenizer(tokenizer: Any, output_dir: str | Path, old_to_new: dict[int, int]) -> Path:
    output = Path(output_dir)
    tokenizer.save_pretrained(output)
    tokenizer_json = output / "tokenizer.json"
    if not tokenizer_json.exists():
        raise VocabRewriteError("Tokenizer did not save tokenizer.json; v1 can only rewrite JSON tokenizers.")
    retained_tokens = rewrite_tokenizer_json(tokenizer_json, old_to_new)
    _rewrite_vocab_json(output / "vocab.json", old_to_new)
    added_tokens_path = output / "added_tokens.json"
    if added_tokens_path.exists():
        added_tokens = json.loads(added_tokens_path.read_text(encoding="utf-8"))
        if isinstance(added_tokens, dict):
            added_tokens_path.write_text(
                json.dumps(_remap_added_tokens(added_tokens, old_to_new), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    _rewrite_tokenizer_config(output / "tokenizer_config.json", old_to_new)
    _rewrite_merges_file(output / "merges.txt", retained_tokens)
    validate_saved_tokenizer(output, len(old_to_new))
    return tokenizer_json


def rewrite_tokenizer_json(path: str | Path, old_to_new: dict[int, int]) -> set[str]:
    tokenizer_path = Path(path)
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model")
    if not isinstance(model, dict):
        raise VocabRewriteError("Unsupported tokenizer.json: missing model object.")

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
        raise VocabRewriteError("Unsupported tokenizer.json: model.vocab must be an object or list.")

    if not model["vocab"]:
        raise VocabRewriteError("Tokenizer rewrite produced an empty vocabulary.")

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
    _remap_json_token_ids(data, old_to_new, path="tokenizer", skip_keys={"vocab", "added_tokens"})
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
        raise VocabRewriteError(f"Pruned tokenizer could not be reloaded: {exc}") from exc
    ids = sorted(int(token_id) for token_id in tokenizer.get_vocab().values())
    expected = list(range(expected_vocab_size))
    if ids != expected:
        raise VocabRewriteError(
            "Pruned tokenizer IDs are not compact and contiguous after reload "
            f"(expected 0..{expected_vocab_size - 1})."
        )


def _rewrite_vocab_json(path: Path, old_to_new: dict[int, int]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VocabRewriteError("Unsupported vocab.json: expected a token-to-ID object.")
    rewritten = {
        token: old_to_new[int(token_id)]
        for token, token_id in sorted(data.items(), key=lambda item: int(item[1]))
        if int(token_id) in old_to_new
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
    _remap_json_token_ids(data, old_to_new, path="tokenizer_config", skip_keys={"added_tokens_decoder"})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _remap_json_token_ids(
    value: Any,
    old_to_new: dict[int, int],
    *,
    path: str,
    skip_keys: set[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in skip_keys:
                continue
            child_path = f"{path}.{key_text}"
            if _is_token_id_key(key_text):
                value[key] = _remap_config_token_id(child, old_to_new, child_path)
            elif key_text == "id" and isinstance(child, int) and child in old_to_new:
                value[key] = old_to_new[child]
            else:
                _remap_json_token_ids(child, old_to_new, path=child_path, skip_keys=skip_keys)
    elif isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int) and value[1] in old_to_new:
            value[1] = old_to_new[value[1]]
            return
        for index, child in enumerate(value):
            _remap_json_token_ids(child, old_to_new, path=f"{path}[{index}]", skip_keys=skip_keys)


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


def _is_token_id_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_token_id") or lowered.endswith("_token_ids")


def _remap_config_token_id(value: Any, old_to_new: dict[int, int], path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, int):
        if value not in old_to_new:
            raise VocabRewriteError(f"Required configuration token {path}={value} was not retained.")
        return old_to_new[value]
    if isinstance(value, (list, tuple)):
        remapped = [_remap_config_token_id(item, old_to_new, path) for item in value]
        return type(value)(remapped)
    return value


def _remap_added_tokens(value: Any, old_to_new: dict[int, int]) -> list[Any] | dict[str, int]:
    if isinstance(value, dict):
        return {
            str(token): old_to_new[int(token_id)]
            for token, token_id in value.items()
            if isinstance(token_id, int) and token_id in old_to_new
        }
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
