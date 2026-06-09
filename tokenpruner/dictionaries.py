from __future__ import annotations

from importlib import resources
from pathlib import Path

from tokenpruner.models import DictionaryMatch

BUILTIN_DICTIONARIES = {
    "english": "english.txt",
    "programming": "programming_terms.txt",
    "linux": "linux_commands.txt",
    "api": "api_library_names.txt",
    "icons": "icon_terms.txt",
}

MATCH_RANKS = {
    "exact": 2.5,
    "prefix": 1.4,
    "suffix": 1.2,
    "substring": 0.9,
    "subword": 0.7,
}


def load_builtin_dictionaries(names: list[str] | None = None) -> dict[str, set[str]]:
    selected = names or list(BUILTIN_DICTIONARIES)
    dictionaries: dict[str, set[str]] = {}
    for name in selected:
        if name not in BUILTIN_DICTIONARIES:
            raise ValueError(f"Unknown built-in dictionary: {name}")
        filename = BUILTIN_DICTIONARIES[name]
        text = resources.files("tokenpruner.resources").joinpath(filename).read_text(encoding="utf-8")
        dictionaries[name] = _parse_terms(text)
    return dictionaries


def load_custom_dictionary(path: str | Path) -> set[str]:
    return _parse_terms(Path(path).read_text(encoding="utf-8"))


def load_dictionaries(custom_paths: list[str] | None = None, builtin_names: list[str] | None = None) -> dict[str, set[str]]:
    dictionaries = load_builtin_dictionaries(builtin_names)
    for path in custom_paths or []:
        dictionaries[Path(path).stem] = load_custom_dictionary(path)
    return dictionaries


def dictionary_matches(text: str, dictionaries: dict[str, set[str]]) -> list[DictionaryMatch]:
    normalized = normalize_token_text(text)
    if not normalized:
        return []

    matches: list[DictionaryMatch] = []
    for dictionary, terms in dictionaries.items():
        for term in terms:
            mode = _match_mode(normalized, term)
            if mode:
                matches.append(
                    DictionaryMatch(
                        dictionary=dictionary,
                        term=term,
                        mode=mode,
                    )
                )
    return sorted(matches, key=lambda match: MATCH_RANKS[match.mode], reverse=True)


def normalize_token_text(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or char in {"_", "-"}).strip("_-")


def _parse_terms(text: str) -> set[str]:
    terms = set()
    for line in text.splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        terms.add(line)
    return terms


def _match_mode(token: str, term: str) -> str | None:
    if token == term:
        return "exact"
    if len(token) >= 2 and term.startswith(token):
        return "subword"
    if len(token) >= 3 and term.endswith(token):
        return "subword"
    if token.startswith(term) and len(term) >= 3:
        return "prefix"
    if token.endswith(term) and len(term) >= 3:
        return "suffix"
    if len(token) >= 3 and token in term:
        return "subword"
    if len(term) >= 3 and term in token:
        return "substring"
    return None
