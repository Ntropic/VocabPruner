from __future__ import annotations

from functools import lru_cache
from importlib import resources
import re
import unicodedata

SYMBOL_PRESET_FILES = {
    "coding": "coding.txt",
    "shell": "shell.txt",
    "json": "json.txt",
    "yaml": "yaml.txt",
    "xml": "xml.txt",
    "markdown": "markdown.txt",
    "mathematics": "mathematics.txt",
    "ascii-punctuation": "ascii-punctuation.txt",
    "emoji": "emoji.txt",
}

SYMBOL_PRESETS: dict[str, set[str]] = {}
SYMBOL_LITERALS: dict[str, set[str]] = {}


def normalize_presets(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    normalized = {value.strip().lower() for value in values if value.strip()}
    if "all" in normalized:
        normalized.remove("all")
        normalized.update(symbol_presets())
    unknown = normalized - set(symbol_presets())
    if unknown:
        raise ValueError(f"Unsupported symbol presets: {', '.join(sorted(unknown))}")
    return normalized


def matching_symbol_presets(text: str, presets: set[str]) -> list[str]:
    matches = []
    for preset in sorted(presets):
        if preset == "emoji":
            if contains_emoji_or_pictograph(text):
                matches.append("emoji")
            continue
        chars = symbol_presets()[preset]
        literals = symbol_literals()[preset]
        if any(char in chars for char in text) or any(_literal_matches(text, literal) for literal in literals):
            matches.append(preset)
    return matches


def char_matches_symbol_preset(char: str, presets: set[str]) -> bool:
    for preset in presets:
        if preset == "emoji":
            if is_emoji_or_pictograph(char):
                return True
            continue
        if char in symbol_presets()[preset]:
            return True
    return False


@lru_cache(maxsize=1)
def symbol_presets() -> dict[str, set[str]]:
    loaded = {name: _load_symbol_preset(filename)[0] for name, filename in SYMBOL_PRESET_FILES.items()}
    SYMBOL_PRESETS.clear()
    SYMBOL_PRESETS.update(loaded)
    return loaded


@lru_cache(maxsize=1)
def symbol_literals() -> dict[str, set[str]]:
    loaded = {name: _load_symbol_preset(filename)[1] for name, filename in SYMBOL_PRESET_FILES.items()}
    SYMBOL_LITERALS.clear()
    SYMBOL_LITERALS.update(loaded)
    return loaded


def _load_symbol_preset(filename: str) -> tuple[set[str], set[str]]:
    text = resources.files("tokenpruner.resources.symbol_presets").joinpath(filename).read_text(encoding="utf-8")
    chars: set[str] = set()
    literals: set[str] = set()
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("literal:"):
            literal = line.removeprefix("literal:").strip().lower()
            if literal:
                literals.add(literal)
            continue
        chars.update(char for char in line if not char.isspace())
    return chars, literals


def _literal_matches(text: str, literal: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(literal)}(?![A-Za-z0-9_])", text.strip().lower()) is not None


symbol_presets()
symbol_literals()


def contains_emoji_or_pictograph(text: str) -> bool:
    return any(is_emoji_or_pictograph(char) for char in text)


def is_emoji_or_pictograph(char: str) -> bool:
    codepoint = ord(char)
    if codepoint in {0xFE0E, 0xFE0F, 0x200D}:
        return True
    ranges = (
        (0x1F000, 0x1FAFF),
        (0x2600, 0x27BF),
        (0x2300, 0x23FF),
    )
    if any(start <= codepoint <= end for start, end in ranges):
        return True
    name = unicodedata.name(char, "")
    return "EMOJI" in name or "DINGBAT" in name
