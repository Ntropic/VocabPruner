from __future__ import annotations

from typing import Any
import json
import re

from vocabpruner.errors import VocabRewriteError
from vocabpruner.tokenization.gguf import GGUFTokenizer


BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def tokenizer_data(tokenizer: Any) -> dict[str, Any]:
    if isinstance(tokenizer, GGUFTokenizer):
        return {
            "model": {"type": "BPE" if tokenizer.model in {"gpt2", ""} else "WordLevel",
                      "vocab": tokenizer.get_vocab(), "merges": tokenizer.merges,
                      "byte_fallback": bool(tokenizer.byte_token_ids)},
            "pre_tokenizer": {"type": "ByteLevel" if tokenizer.model == "gpt2" else ""},
            "added_tokens": [{"id": token_id, "special": True} for token_id in tokenizer.all_special_ids],
        }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    return json.loads(backend.to_str()) if backend is not None else {}


def model_vocabulary(data: dict[str, Any]) -> dict[str, int]:
    model = data.get("model", {})
    if model.get("type") not in {"BPE", "Unigram", "WordPiece", "WordLevel"}:
        raise VocabRewriteError("Unsupported tokenizer format; expected BPE, Unigram, WordPiece, or WordLevel tokenizer.json.")
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return vocab
    if isinstance(vocab, list) and model["type"] == "Unigram":
        return {entry[0]: index for index, entry in enumerate(vocab)}
    raise VocabRewriteError("Unsupported tokenizer.json: model.vocab must be an object or Unigram list.")


def byte_level_token_ids(data: dict[str, Any]) -> set[int]:
    def has_byte_level(component: Any) -> bool:
        if isinstance(component, dict):
            return component.get("type") == "ByteLevel" or any(has_byte_level(value) for value in component.values())
        if isinstance(component, list):
            return any(has_byte_level(value) for value in component)
        return False

    if not has_byte_level(data.get("pre_tokenizer")):
        return set()
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = model_vocabulary(data)
    alphabet = set(ByteLevel.alphabet())
    missing = alphabet - vocab.keys()
    if missing:
        raise VocabRewriteError(f"Byte-level tokenizer is missing {len(missing)} required byte tokens.")
    return {int(vocab[token]) for token in alphabet}


def required_byte_token_ids(data: dict[str, Any]) -> set[int]:
    model = data.get("model", {})
    vocab = model_vocabulary(data)
    byte_ids = byte_level_token_ids(data)
    if model.get("byte_fallback"):
        required = {f"<0x{value:02X}>" for value in range(256)}
        if required - vocab.keys():
            raise VocabRewriteError("Byte-fallback tokenizer must contain all 256 fallback tokens.")
    byte_ids.update(token_id for token, token_id in vocab.items() if BYTE_FALLBACK_RE.fullmatch(token))
    return byte_ids


def complete_token_ids(data: dict[str, Any], retained_ids: list[int]) -> list[int]:
    byte_ids = required_byte_token_ids(data)
    if not retained_ids:
        return sorted(set(retained_ids))
    vocab = model_vocabulary(data)
    dependencies: dict[int, set[int]] = {}
    for merge in data["model"].get("merges", []):
        parts = merge.split() if isinstance(merge, str) else merge
        if not isinstance(parts, (list, tuple)) or len(parts) != 2:
            raise VocabRewriteError("Source tokenizer has a malformed BPE merge.")
        left, right = parts
        prefix = data["model"].get("continuing_subword_prefix") or ""
        merged = left + (right[len(prefix):] if prefix and right.startswith(prefix) else right)
        if left not in vocab or right not in vocab or merged not in vocab:
            raise VocabRewriteError("Source tokenizer has a merge referencing missing vocabulary entries.")
        dependencies.setdefault(vocab[merged], set()).update((vocab[left], vocab[right]))
    retained = set(retained_ids) | byte_ids
    retained.update(int(token["id"]) for token in data.get("added_tokens", []) if token.get("special"))
    unknown = data["model"].get("unk_token")
    if unknown in vocab:
        retained.add(vocab[unknown])
    if isinstance(data["model"].get("unk_id"), int):
        retained.add(data["model"]["unk_id"])
    pending = list(retained)
    while pending:
        token_id = pending.pop()
        required = dependencies.get(token_id, set()) - retained
        retained.update(required)
        pending.extend(required)
    return sorted(retained)


def complete_retained_token_ids(tokenizer: Any, retained_ids: list[int]) -> list[int]:
    if tokenizer is None:
        return retained_ids
    return complete_token_ids(tokenizer_data(tokenizer), retained_ids)


def is_byte_fallback_token(raw_token: str) -> bool:
    if BYTE_FALLBACK_RE.match(raw_token):
        return True
    if len(raw_token) == 1 and 0 <= ord(raw_token) <= 255 and raw_token.isprintable() is False:
        return True
    return False


def merge_is_retained(merge: Any, retained_tokens: set[str], continuing_subword_prefix: str = "") -> bool:
    if isinstance(merge, str):
        parts = merge.split()
    elif isinstance(merge, list):
        parts = [str(part) for part in merge]
    else:
        return False
    if len(parts) != 2:
        return False
    left, right = parts
    suffix = right[len(continuing_subword_prefix):] if continuing_subword_prefix and right.startswith(continuing_subword_prefix) else right
    return left in retained_tokens and right in retained_tokens and left + suffix in retained_tokens
