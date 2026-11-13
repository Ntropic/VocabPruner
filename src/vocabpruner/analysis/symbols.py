from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import resources
import json
import re


SYNTAX_LIST_PACKAGE = "vocabpruner.resources.syntax_lists"


SYMBOL_LIST_PACKAGE = "vocabpruner.resources.symbol_lists"


@dataclass(frozen=True)
class CharacterList:
    characters: frozenset[str]
    words: frozenset[str]
    ranges: tuple[tuple[int, int], ...]


def normalize_syntax_lists(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    return _normalize_lists(values, syntax_lists(), "syntax")


def normalize_symbol_lists(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    return _normalize_lists(values, symbol_lists(), "symbol")


def _normalize_lists(values, available: dict[str, CharacterList], kind: str) -> set[str]:
    normalized = {value.strip().lower() for value in values if value.strip()}
    if "all" in normalized:
        normalized.remove("all")
        normalized.update(available)
    unknown = normalized - set(available)
    if unknown:
        raise ValueError(f"Unsupported {kind} lists: {', '.join(sorted(unknown))}")
    return normalized


def matching_syntax_lists(text: str, selected: set[str]) -> list[str]:
    return _matching_lists(text, selected, syntax_lists())


def matching_syntax_word_lists(text: str, selected: set[str]) -> list[str]:
    return [
        name
        for name in sorted(selected)
        if any(_word_matches(text, word) for word in syntax_lists()[name].words)
    ]


def matching_symbol_lists(text: str, selected: set[str]) -> list[str]:
    return _matching_lists(text, selected, symbol_lists())


def _matching_lists(text: str, selected: set[str], available: dict[str, CharacterList]) -> list[str]:
    return [
        name
        for name in sorted(selected)
        if any(_char_matches(char, available[name]) for char in text)
        or any(_word_matches(text, word) for word in available[name].words)
    ]


def char_matches_list(char: str, syntax: set[str], symbols: set[str]) -> bool:
    return any(_char_matches(char, syntax_lists()[name]) for name in syntax) or any(
        _char_matches(char, symbol_lists()[name]) for name in symbols
    )


def _char_matches(char: str, definition: CharacterList) -> bool:
    codepoint = ord(char)
    return char in definition.characters or any(start <= codepoint <= end for start, end in definition.ranges)


def _word_matches(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text.strip().lower()) is not None


def compact_list_values(values: set[str], available: dict[str, CharacterList]) -> list[str]:
    return ["all"] if values == set(available) else sorted(values)


def syntax_list_fingerprint(selected: set[str]) -> str:
    return _list_fingerprint(SYNTAX_LIST_PACKAGE, selected)


def symbol_list_fingerprint(selected: set[str]) -> str:
    return _list_fingerprint(SYMBOL_LIST_PACKAGE, selected)


def _list_fingerprint(package: str, selected: set[str]) -> str:
    digest = sha256()
    root = resources.files(package)
    for name in sorted(selected):
        digest.update(name.encode())
        digest.update(root.joinpath(f"{name}.json").read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def syntax_lists() -> dict[str, CharacterList]:
    return _load_lists(SYNTAX_LIST_PACKAGE)


@lru_cache(maxsize=1)
def symbol_lists() -> dict[str, CharacterList]:
    return _load_lists(SYMBOL_LIST_PACKAGE)


def _load_lists(package: str) -> dict[str, CharacterList]:
    root = resources.files(package)
    return {
        item.name.removesuffix(".json"): _load_list(item)
        for item in sorted(root.iterdir(), key=lambda value: value.name)
        if item.is_file() and item.name.endswith(".json")
    }


def _load_list(path) -> CharacterList:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"characters", "words", "ranges"}:
        raise ValueError(f"{path.name} must contain characters, words, and ranges")
    characters = data["characters"]
    words = data["words"]
    ranges = data["ranges"]
    if not isinstance(characters, str) or not isinstance(words, list) or not all(isinstance(word, str) for word in words):
        raise ValueError(f"Invalid characters or words in {path.name}")
    parsed_ranges = tuple(_parse_range(item, path.name) for item in ranges)
    return CharacterList(frozenset(characters), frozenset(word.lower() for word in words), parsed_ranges)


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
