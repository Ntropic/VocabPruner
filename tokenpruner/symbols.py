from __future__ import annotations

import unicodedata

SYMBOL_PRESETS: dict[str, set[str]] = {
    "coding": set("{}[]()<>+-=*/%&|^~!?:;.,_#@$\\'\"`"),
    "shell": set("-_./~$*?[]{}|&;<>`'\"\\!"),
    "json": set("{}[]:,.\"\\-0123456789truefalsenull"),
    "yaml": set("-_:[]{}.,#&*!|>'\"%@`"),
    "xml": set("<>/=\"'!?-_:.;#&"),
    "markdown": set("#*_`[]()!>-+.|:\\"),
    "mathematics": set("+-=*/×÷±≈≠≤≥∞∑∏√∫∂∆∇πµ^%<>|"),
    "ascii-punctuation": set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
    "emoji": set(),
}


def normalize_presets(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    normalized = {value.strip().lower() for value in values if value.strip()}
    if "all" in normalized:
        normalized.remove("all")
        normalized.update(SYMBOL_PRESETS)
    unknown = normalized - set(SYMBOL_PRESETS)
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
        chars = SYMBOL_PRESETS[preset]
        if any(char in chars for char in text):
            matches.append(preset)
    return matches


def char_matches_symbol_preset(char: str, presets: set[str]) -> bool:
    for preset in presets:
        if preset == "emoji":
            if is_emoji_or_pictograph(char):
                return True
            continue
        if char in SYMBOL_PRESETS[preset]:
            return True
    return False


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
