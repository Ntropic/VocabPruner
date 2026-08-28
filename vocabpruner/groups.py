from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json

SELECTION_GROUP_PACKAGE = "vocabpruner.resources.selection_groups"


@dataclass(frozen=True)
class SelectionGroup:
    keep_scripts: tuple[str, ...]
    syntax_lists: tuple[str, ...]
    symbol_lists: tuple[str, ...]


@lru_cache(maxsize=1)
def selection_groups() -> dict[str, SelectionGroup]:
    root = resources.files(SELECTION_GROUP_PACKAGE)
    return {
        item.name.removesuffix(".json"): _load_group(item)
        for item in sorted(root.iterdir(), key=lambda value: value.name)
        if item.is_file() and item.name.endswith(".json")
    }


def _load_group(path) -> SelectionGroup:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = {"keep_scripts", "syntax_lists", "symbol_lists"}
    if set(data) != keys or any(
        not isinstance(data[key], list) or not all(isinstance(value, str) for value in data[key])
        for key in keys
    ):
        raise ValueError(f"{path.name} must contain string lists for {', '.join(sorted(keys))}")
    return SelectionGroup(
        tuple(data["keep_scripts"]),
        tuple(data["syntax_lists"]),
        tuple(data["symbol_lists"]),
    )
