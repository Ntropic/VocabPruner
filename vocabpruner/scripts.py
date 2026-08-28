from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import resources
import json
import unicodedata

from vocabpruner.models import ScriptProfile

SCRIPT_LIST_PACKAGE = "vocabpruner.resources.script_lists"


@dataclass(frozen=True)
class ScriptList:
    characters: frozenset[str]
    categories: frozenset[str]
    ranges: tuple[tuple[int, int], ...]


def scripts_for_char(char: str) -> frozenset[str]:
    codepoint = ord(char)
    category = unicodedata.category(char)
    matches = frozenset(
        name
        for name, definition in script_lists().items()
        if (
            char in definition.characters
            or any(category.startswith(value) for value in definition.categories)
            or any(start <= codepoint <= end for start, end in definition.ranges)
        )
    )
    return matches or frozenset({"Unknown"})


def script_profile(text: str) -> ScriptProfile:
    counts: dict[str, int] = {}
    meaningful = 0
    total = 0
    for char in text:
        if char.isspace():
            continue
        scripts = scripts_for_char(char)
        total += 1
        for script in scripts:
            counts[script] = counts.get(script, 0) + 1
        if scripts - {"Common", "Inherited"}:
            meaningful += 1
    low_confidence = meaningful <= 2 and total <= 3
    return ScriptProfile(counts=counts, total_meaningful=meaningful, low_confidence=low_confidence)


def normalize_scripts(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    available = script_lists()
    canonical = {name.lower(): name for name in available}
    normalized: set[str] = set()
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value.lower() == "all":
            normalized.update(available)
        elif value.lower() in canonical:
            normalized.add(canonical[value.lower()])
        else:
            normalized.add(value)
    unknown = normalized - set(available)
    if unknown:
        raise ValueError(f"Unsupported scripts: {', '.join(sorted(unknown))}")
    return normalized


def compact_script_values(values: set[str]) -> list[str]:
    return ["all"] if values == set(script_lists()) else sorted(values)


def script_list_fingerprint(selected: set[str]) -> str:
    digest = sha256()
    root = resources.files(SCRIPT_LIST_PACKAGE)
    for name in sorted(selected):
        digest.update(name.encode())
        digest.update(root.joinpath(f"{name}.json").read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def script_lists() -> dict[str, ScriptList]:
    root = resources.files(SCRIPT_LIST_PACKAGE)
    return {
        item.name.removesuffix(".json"): _load_script_list(item)
        for item in sorted(root.iterdir(), key=lambda value: value.name)
        if item.is_file() and item.name.endswith(".json")
    }


def _load_script_list(path) -> ScriptList:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"characters", "categories", "ranges"}:
        raise ValueError(f"{path.name} must contain characters, categories, and ranges")
    characters = data["characters"]
    categories = data["categories"]
    ranges = data["ranges"]
    if (
        not isinstance(characters, str)
        or not isinstance(categories, list)
        or not all(isinstance(category, str) and category for category in categories)
        or not isinstance(ranges, list)
    ):
        raise ValueError(f"Invalid script list in {path.name}")
    return ScriptList(
        frozenset(characters),
        frozenset(categories),
        tuple(_parse_range(value, path.name) for value in ranges),
    )


def _parse_range(value, filename: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid range in {filename}: {value!r}")
    try:
        start, end = (int(item.removeprefix("U+").removeprefix("u+"), 16) for item in value)
    except ValueError as exc:
        raise ValueError(f"Invalid range in {filename}: {value!r}") from exc
    if not 0 <= start <= end <= 0x10FFFF:
        raise ValueError(f"Invalid range in {filename}: {value!r}")
    return start, end
