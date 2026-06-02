from __future__ import annotations

import unicodedata

from tokenpruner.models import ScriptProfile

SUPPORTED_SCRIPTS = {
    "Latin",
    "Common",
    "Inherited",
    "Han",
    "Hiragana",
    "Katakana",
    "Hangul",
    "Cyrillic",
    "Arabic",
    "Hebrew",
    "Devanagari",
    "Thai",
    "Greek",
}

SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Latin", 0x0041, 0x007A),
    ("Latin", 0x00C0, 0x024F),
    ("Latin", 0x1E00, 0x1EFF),
    ("Greek", 0x0370, 0x03FF),
    ("Greek", 0x1F00, 0x1FFF),
    ("Cyrillic", 0x0400, 0x052F),
    ("Hebrew", 0x0590, 0x05FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F),
    ("Arabic", 0x08A0, 0x08FF),
    ("Devanagari", 0x0900, 0x097F),
    ("Thai", 0x0E00, 0x0E7F),
    ("Han", 0x3400, 0x4DBF),
    ("Han", 0x4E00, 0x9FFF),
    ("Han", 0xF900, 0xFAFF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Katakana", 0x31F0, 0x31FF),
    ("Hangul", 0xAC00, 0xD7AF),
    ("Hangul", 0x1100, 0x11FF),
    ("Hangul", 0x3130, 0x318F),
)


def script_for_char(char: str) -> str:
    codepoint = ord(char)
    category = unicodedata.category(char)
    if category.startswith("M"):
        return "Inherited"
    for script, start, end in SCRIPT_RANGES:
        if start <= codepoint <= end:
            return script
    if category[0] in {"P", "S", "Z", "N", "C"}:
        return "Common"
    return "Common"


def script_profile(text: str) -> ScriptProfile:
    counts: dict[str, int] = {}
    meaningful = 0
    for char in text:
        if char.isspace():
            continue
        script = script_for_char(char)
        counts[script] = counts.get(script, 0) + 1
        if script not in {"Common", "Inherited"}:
            meaningful += 1
    total = sum(counts.values())
    low_confidence = meaningful <= 2 and total <= 3
    return ScriptProfile(counts=counts, total_meaningful=total, low_confidence=low_confidence)


def normalize_scripts(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    normalized = {value.strip().title() for value in values if value.strip()}
    unknown = normalized - SUPPORTED_SCRIPTS
    if unknown:
        raise ValueError(f"Unsupported scripts: {', '.join(sorted(unknown))}")
    return normalized
