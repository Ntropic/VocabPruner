from __future__ import annotations

import re
import os
from collections.abc import Mapping
from typing import Any

from tokenpruner.hf_cache import mark_analysis_download
from tokenpruner.models import TokenRecord
from tokenpruner.paths import HF_MODELS_DIR

BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


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
