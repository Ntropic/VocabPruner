from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from vocabpruner.errors import VocabRewriteError
from vocabpruner.mapping import remap_added_tokens, remap_json_token_ids
from vocabpruner.storage.gguf import gguf_field, read_gguf
from vocabpruner.tokenization.gguf import GGUFTokenizer
from vocabpruner.tokenization.loading import special_token_ids
from vocabpruner.tokenization.preservation import (
    merge_is_retained,
    complete_retained_token_ids,
    required_byte_token_ids,
    tokenizer_data,
)


def validate_saved_tokenizer(
    output_dir: str | Path,
    expected_vocab_size: int,
    source_tokenizer: Any = None,
    old_to_new: dict[int, int] | None = None,
) -> None:
    (Path(output_dir) / "tokenizer_validation.json").unlink(missing_ok=True)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise VocabRewriteError("Transformers is required to validate the saved tokenizer.") from exc
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
    if source_tokenizer is None or old_to_new is None:
        raise VocabRewriteError("Source tokenizer and token map are required for post-pruning validation.")
    report = validate_tokenizer_vocabulary(source_tokenizer, tokenizer, old_to_new)
    (Path(output_dir) / "tokenizer_validation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_tokenizer_vocabulary(source: Any, saved: Any, old_to_new: dict[int, int]) -> dict[str, Any]:
    source_data = tokenizer_data(source)
    saved_data = tokenizer_data(saved)
    required = set(complete_retained_token_ids(source, list(old_to_new)))
    if required - old_to_new.keys():
        raise VocabRewriteError("Token map omits required byte, special, unknown, or merge-dependency tokens.")
    source_vocab = source.get_vocab()
    expected_vocab = {token: old_to_new[token_id] for token, token_id in source_vocab.items() if token_id in old_to_new}
    if saved.get_vocab() != expected_vocab or sorted(expected_vocab.values()) != list(range(len(old_to_new))):
        raise VocabRewriteError("Saved vocabulary does not match the old-to-new token map.")
    source_special = special_token_ids(source)
    saved_special = special_token_ids(saved)
    for token, new_id in expected_vocab.items():
        old_id = source_vocab[token]
        if (old_id in source_special) != (new_id in saved_special):
            raise VocabRewriteError(f"Special-token status changed for {token!r} ({old_id} -> {new_id}).")
        original = source.decode([old_id], clean_up_tokenization_spaces=False)
        decoded = saved.decode([new_id], clean_up_tokenization_spaces=False)
        if decoded != original:
            raise VocabRewriteError(f"Decoded token changed for {token!r} ({old_id} -> {new_id}): {original!r} -> {decoded!r}.")
    for key in ("normalizer", "pre_tokenizer", "decoder"):
        if source_data.get(key) != saved_data.get(key):
            raise VocabRewriteError(f"Tokenizer {key} changed during pruning.")
    original_model = dict(source_data["model"])
    saved_model = dict(saved_data["model"])
    original_vocab = original_model.pop("vocab")
    saved_vocab = saved_model.pop("vocab")
    original_merges = original_model.pop("merges", [])
    saved_merges = saved_model.pop("merges", [])
    if isinstance(original_vocab, list):
        expected_entries = [entry for token_id, entry in enumerate(original_vocab) if token_id in old_to_new]
        if saved_vocab != expected_entries:
            raise VocabRewriteError("Unigram token strings or scores changed during pruning.")
    if isinstance(original_model.get("unk_id"), int):
        original_model["unk_id"] = old_to_new[original_model["unk_id"]]
    if original_model != saved_model:
        raise VocabRewriteError("Tokenizer model settings changed during pruning.")
    prefix = original_model.get("continuing_subword_prefix") or ""
    expected_tokens = set(expected_vocab)
    expected_merges = [merge for merge in original_merges if merge_is_retained(merge, expected_tokens, prefix)]
    normalize_merges = lambda merges: [merge.split() if isinstance(merge, str) else list(merge) for merge in merges]
    if normalize_merges(saved_merges) != normalize_merges(expected_merges):
        raise VocabRewriteError("Retained BPE merge rules or their order changed during pruning.")
    if remap_added_tokens(source_data.get("added_tokens", []), old_to_new) != saved_data.get("added_tokens", []):
        raise VocabRewriteError("Retained added-token attributes changed during pruning.")
    original_processor = json.loads(json.dumps(source_data.get("post_processor")))
    remap_json_token_ids(original_processor, old_to_new, path="post_processor", skip_keys=set())
    if original_processor != saved_data.get("post_processor"):
        raise VocabRewriteError("Tokenizer post-processor changed during pruning.")
    return {"status": "passed", "tokenizer_type": saved_model["type"],
            "retained_tokens_checked": len(old_to_new), "byte_tokens_preserved": len(required_byte_token_ids(saved_data)),
            "checks": ["vocabulary and ID map", "individual token decoding", "special and added tokens",
                       "model settings and Unigram scores", "BPE dependencies and merge order", "text processing settings"],
            "scope": "tokenizer vocabulary integrity only; no model inference or arbitrary-text guarantee"}


def validate_gguf_tokenizer(source: str | Path, output: str | Path, retained_ids: list[int]) -> None:
    import json

    from vocabpruner.tokenization.validation import validate_tokenizer_vocabulary

    report_path = Path(output).with_suffix(".tokenizer_validation.json")
    report_path.unlink(missing_ok=True)
    original = GGUFTokenizer(source)
    saved = GGUFTokenizer(output)
    if saved.tokens != [original.tokens[token_id] for token_id in retained_ids]:
        raise ValueError("Saved GGUF vocabulary does not match the retained token map.")
    mapping = {old_id: new_id for new_id, old_id in enumerate(retained_ids)}
    report = validate_tokenizer_vocabulary(original, saved, mapping)
    report["tokenizer_type"] = f"GGUF/{saved.model or 'unspecified'}"
    source_reader = read_gguf(source)
    saved_reader = read_gguf(output)
    for key in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
        original_values = gguf_field(source_reader, key)
        if original_values is not None and gguf_field(saved_reader, key) != [original_values[index] for index in retained_ids]:
            raise ValueError(f"Retained GGUF token attributes changed: {key}.")
    report["scope"] = "GGUF vocabulary, metadata and wrapper decoding only; no model inference or native-runtime tokenization check"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
