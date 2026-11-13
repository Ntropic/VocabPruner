from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from vocabpruner.mapping import is_token_id_key, remap_config_token_id


def write_pruned_config(config: Any, path: Path, old_to_new: dict[int, int], original_vocab_size: int) -> None:
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    remap_model_config(data, old_to_new, original_vocab_size)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remap_generation_config(path: Path, old_to_new: dict[int, int], original_vocab_size: int) -> None:
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    remap_model_config(data, old_to_new, original_vocab_size)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            if is_token_id_key(key_text):
                value[key] = remap_config_token_id(child, old_to_new, child_path)
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
    attribute_names = set(vars(value))
    attribute_names.update(
        key
        for key in dir(value)
        if not key.startswith("_") and (key == "vocab_size" or is_token_id_key(key))
    )
    for key in sorted(attribute_names):
        try:
            child = getattr(value, key)
        except Exception:
            continue
        child_path = f"{path}.{key}"
        if is_token_id_key(key):
            setattr(value, key, remap_config_token_id(child, old_to_new, child_path))
        elif key == "vocab_size" and child == original_vocab_size:
            setattr(value, key, len(old_to_new))
        elif not key.startswith("_"):
            _remap_config_value(child, old_to_new, original_vocab_size, path=child_path, seen=seen)
