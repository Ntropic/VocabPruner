from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any
import json


@lru_cache(maxsize=1)
def codebook_rules() -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    tensors: list[tuple[str, int]] = []
    config_token_ids: set[str] = set()
    root = resources.files("vocabpruner.resources.codebooks")
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.name.endswith(".json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tensors.extend((rule["suffix"], rule["axis"]) for rule in data["tensors"])
        config_token_ids.update(data["config_token_ids"])
    return tuple(tensors), tuple(sorted(config_token_ids))


def codebook_tensor_axis(name: str, shape: Any, vocab_size: int) -> int | None:
    for suffix, axis in codebook_rules()[0]:
        if (name == suffix or name.endswith(f".{suffix}")) and axis < len(shape) and shape[axis] == vocab_size:
            return axis
    return None


def codebook_config_ids(config: Any) -> set[int]:
    data = config.to_dict() if hasattr(config, "to_dict") else config
    found: set[int] = set()
    for dotted_path in codebook_rules()[1]:
        value = data
        for key in dotted_path.split("."):
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            values = value if isinstance(value, (list, tuple)) else [value]
            found.update(item for item in values if isinstance(item, int) and not isinstance(item, bool))
    return found
