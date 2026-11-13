from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import os

from vocabpruner.models import TokenRecord
from vocabpruner.storage.gguf import resolve_gguf
from vocabpruner.storage.hub import mark_analysis_download
from vocabpruner.storage.paths import hf_models_dir
from vocabpruner.tokenization.gguf import GGUFTokenizer
from vocabpruner.tokenization.preservation import (
    byte_level_token_ids,
    is_byte_fallback_token,
    tokenizer_data,
)


def load_tokenizer(model: str, trust_remote_code: bool = False, **kwargs: Any) -> Any:
    gguf = resolve_gguf(model)
    if gguf is not None:
        return GGUFTokenizer(gguf)
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
        cache_dir=hf_models_dir(),
        **kwargs,
    )
    mark_analysis_download(model, trust_remote_code)
    return tokenizer


def tokenizer_records(tokenizer: Any) -> list[TokenRecord]:
    vocab = tokenizer.get_vocab()
    byte_ids = byte_level_token_ids(tokenizer_data(tokenizer))
    byte_ids.update(getattr(tokenizer, "byte_token_ids", []))
    special_ids = special_token_ids(tokenizer)
    records: list[TokenRecord] = []
    for raw_token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        decoded = decode_token(tokenizer, token_id, raw_token)
        records.append(
            TokenRecord(
                token_id=int(token_id),
                raw_token=str(raw_token),
                decoded_text=decoded,
                is_special=int(token_id) in special_ids or str(raw_token) in _special_tokens(tokenizer),
                is_byte_fallback=int(token_id) in byte_ids or is_byte_fallback_token(str(raw_token)),
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


def special_token_ids(tokenizer: Any) -> set[int]:
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
