from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from vocabpruner.errors import VocabRewriteError
from vocabpruner.mapping import remap_added_tokens, remap_config_token_id, remap_json_token_ids
from vocabpruner.tokenization.preservation import (
    merge_is_retained,
    complete_retained_token_ids,
    complete_token_ids,
    model_vocabulary,
    tokenizer_data,
)
from vocabpruner.tokenization.validation import validate_saved_tokenizer


def save_remapped_tokenizer(tokenizer: Any, output_dir: str | Path, old_to_new: dict[int, int]) -> Path:
    (Path(output_dir) / "tokenizer_validation.json").unlink(missing_ok=True)
    if set(complete_retained_token_ids(tokenizer, list(old_to_new))) - old_to_new.keys():
        raise VocabRewriteError("Token map omits required byte tokens or BPE merge dependencies.")
    output = Path(output_dir)
    tokenizer.save_pretrained(output)
    tokenizer_json = output / "tokenizer.json"
    if not tokenizer_json.exists():
        raise VocabRewriteError("Tokenizer did not save tokenizer.json; only JSON tokenizer exports are supported.")
    retained_tokens = rewrite_tokenizer_json(tokenizer_json, old_to_new)
    _rewrite_vocab_json(output / "vocab.json", old_to_new)
    added_tokens_path = output / "added_tokens.json"
    if added_tokens_path.exists():
        added_tokens = json.loads(added_tokens_path.read_text(encoding="utf-8"))
        if isinstance(added_tokens, dict):
            added_tokens_path.write_text(
                json.dumps(remap_added_tokens(added_tokens, old_to_new), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    _rewrite_tokenizer_config(output / "tokenizer_config.json", old_to_new)
    source_data = tokenizer_data(tokenizer)
    prefix = source_data.get("model", {}).get("continuing_subword_prefix") or ""
    _rewrite_merges_file(output / "merges.txt", retained_tokens, prefix)
    vocab_text = output / "vocab.txt"
    if source_data.get("model", {}).get("type") == "WordPiece" and vocab_text.exists():
        vocab = model_vocabulary(source_data)
        ordered = sorted(vocab, key=vocab.get)
        vocab_text.write_text("".join(token + "\n" for token in ordered if vocab[token] in old_to_new), encoding="utf-8")
    for filename in ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model"):
        (output / filename).unlink(missing_ok=True)
    validate_saved_tokenizer(output, len(old_to_new), source_tokenizer=tokenizer, old_to_new=old_to_new)
    return tokenizer_json


def rewrite_tokenizer_json(path: str | Path, old_to_new: dict[int, int]) -> set[str]:
    tokenizer_path = Path(path)
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model")
    if not isinstance(model, dict):
        raise VocabRewriteError("Unsupported tokenizer.json: missing model object.")
    model_vocabulary(data)
    if old_to_new != {old_id: new_id for new_id, old_id in enumerate(sorted(old_to_new))}:
        raise VocabRewriteError("Token map must compact IDs in source vocabulary order.")
    required_ids = set(complete_token_ids(data, list(old_to_new)))
    if required_ids - old_to_new.keys():
        raise VocabRewriteError("Token map omits required byte, special, unknown, or BPE dependency tokens; re-prune the source model.")

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
    if isinstance(model.get("unk_id"), int):
        model["unk_id"] = remap_config_token_id(model["unk_id"], old_to_new, "model.unk_id")

    retained_tokens = (
        set(model["vocab"])
        if isinstance(model["vocab"], dict)
        else {str(entry[0]) for entry in model["vocab"] if isinstance(entry, list) and entry}
    )
    if isinstance(model.get("merges"), list):
        model["merges"] = [
            merge
            for merge in model["merges"]
            if merge_is_retained(merge, retained_tokens, model.get("continuing_subword_prefix") or "")
        ]
    data["added_tokens"] = remap_added_tokens(data.get("added_tokens", []), old_to_new)
    remap_json_token_ids(data, old_to_new, path="tokenizer", skip_keys={"vocab", "added_tokens"})
    tokenizer_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return retained_tokens


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
    remap_json_token_ids(data, old_to_new, path="tokenizer_config", skip_keys={"added_tokens_decoder"})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rewrite_merges_file(path: Path, retained_tokens: set[str], continuing_subword_prefix: str = "") -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [line for line in lines if line.startswith("#")]
    merges = [line for line in lines if line and not line.startswith("#") and merge_is_retained(line, retained_tokens, continuing_subword_prefix)]
    path.write_text("\n".join(header + merges) + "\n", encoding="utf-8")
