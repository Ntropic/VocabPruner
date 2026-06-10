from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from tokenpruner.dictionaries import dictionary_matches, load_dictionaries
from tokenpruner.models import TokenRecord, TokenScore
from tokenpruner.scripts import normalize_scripts, script_for_char, script_profile
from tokenpruner.symbols import char_matches_symbol_preset, matching_symbol_presets, normalize_presets
from tokenpruner.wordfreq_support import wordfreq_matches

ESCAPE_RE = re.compile(r"""\\(?:[nrtbfav\\"']|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})""")
IDENTIFIER_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+")


@dataclass(frozen=True)
class ScoreConfig:
    keep_scripts: set[str] = field(default_factory=lambda: {"Latin", "Common", "Inherited"})
    symbol_presets: set[str] = field(default_factory=lambda: {"coding", "ascii-punctuation"})
    languages: tuple[str, ...] = ("en",)
    wordfreq_min_zipf: float = 4.0
    wordfreq_top_pct: float | None = None
    custom_dictionaries: tuple[str, ...] = ()
    builtin_dictionaries: tuple[str, ...] | None = None

    @classmethod
    def from_cli(
        cls,
        keep_scripts: list[str] | None = None,
        symbol_presets: list[str] | None = None,
        languages: list[str] | None = None,
        wordfreq_min_zipf: float | None = None,
        wordfreq_top_pct: float | None = None,
        dictionaries: list[str] | None = None,
    ) -> "ScoreConfig":
        if wordfreq_min_zipf is not None and wordfreq_top_pct is not None:
            raise ValueError("Use either --wordfreq-min-zipf or --wordfreq-top-pct, not both.")
        if wordfreq_min_zipf is not None and wordfreq_min_zipf < 0:
            raise ValueError("wordfreq_min_zipf must be non-negative")
        if wordfreq_top_pct is not None and not 0 < wordfreq_top_pct <= 100:
            raise ValueError("wordfreq_top_pct must be greater than 0 and at most 100")
        return cls(
            keep_scripts=normalize_scripts(keep_scripts or ["Latin", "Common", "Inherited"]),
            symbol_presets=normalize_presets(symbol_presets or ["coding", "ascii-punctuation"]),
            languages=tuple(languages or ["en"]),
            wordfreq_min_zipf=4.0 if wordfreq_min_zipf is None and wordfreq_top_pct is None else (wordfreq_min_zipf or 0.0),
            wordfreq_top_pct=wordfreq_top_pct,
            custom_dictionaries=tuple(dictionaries or []),
        )


def score_token(record: TokenRecord, config: ScoreConfig, dictionaries: dict[str, set[str]] | None = None) -> TokenScore:
    text = record.decoded_text or record.raw_token
    profile = script_profile(text)
    dictionaries = dictionaries if dictionaries is not None else load_dictionaries(
        list(config.custom_dictionaries),
        list(config.builtin_dictionaries) if config.builtin_dictionaries is not None else None,
    )

    explanations: list[str] = []

    structural = is_structural_token(text)
    protected = record.is_special or record.is_byte_fallback or structural
    if record.is_special:
        explanations.append("special token is always preserved")
    if record.is_byte_fallback:
        explanations.append("byte fallback token is always preserved")
    if structural:
        explanations.append("decoded whitespace/control token is always preserved")

    segments = token_segments(text)
    presets = matching_symbol_presets(text, config.symbol_presets)
    gate = character_gate(text, config, segments)

    if not protected and gate.blocked:
        explanations.append(f"character gate blocked disallowed character families: {', '.join(gate.blocked)}")
    elif not protected:
        explanations.append("character gate passed")

    word_segments = language_word_segments(segments)
    evidence_segments = language_evidence_segments(word_segments, dictionaries)
    has_escape_segments = any(segment.escaped for segment in segments)
    has_symbol_evidence = bool(presets)

    if presets and (protected or not gate.blocked):
        explanations.append(f"symbol/icon presets matched: {', '.join(sorted(set(presets)))}")

    dict_matches = []
    wf_matches = []
    segment_evidence: list[tuple[TokenSegment, bool]] = []
    if not protected and not gate.blocked and evidence_segments:
        for segment in evidence_segments:
            segment_dict_matches = dictionary_matches(segment.text, dictionaries)
            segment_wf_matches = wordfreq_matches(
                segment.text,
                list(config.languages),
                min_zipf=config.wordfreq_min_zipf,
                top_pct=config.wordfreq_top_pct,
            )
            dict_matches.extend(segment_dict_matches)
            wf_matches.extend(segment_wf_matches)
            segment_evidence.append(
                (
                    segment,
                    has_language_evidence(segment.text, segment_dict_matches, segment_wf_matches),
                )
            )
    if dict_matches:
        top = ", ".join(f"{match.dictionary}:{match.term}/{match.mode}" for match in dict_matches[:3])
        explanations.append(f"dictionary evidence matched: {top}")

    if wf_matches:
        top = ", ".join(f"{match.language}:{match.term}/{match.mode}" for match in wf_matches[:3])
        explanations.append(f"wordfreq evidence matched: {top}")

    if not protected and not gate.blocked:
        if word_segments:
            missing = [
                segment.text
                for segment, accepted in segment_evidence
                if not accepted
            ]
            if missing:
                explanations.append(f"word segment(s) lack language evidence: {', '.join(missing[:5])}")
        elif has_symbol_evidence or has_escape_segments:
            if has_escape_segments and not has_symbol_evidence:
                explanations.append("escape sequence retained as structural syntax")
            else:
                explanations.append("symbol-only token retained by selected symbol presets")
        else:
            explanations.append("no language word segments or selected symbol evidence")

    all_words_accepted = bool(segment_evidence) and all(accepted for _, accepted in segment_evidence)
    status = decision_tree_status(
        protected=protected,
        gate_blocked=bool(gate.blocked),
        has_word_segments=bool(word_segments),
        all_words_accepted=all_words_accepted,
    )
    explanations.append(_decision_explanation(status, protected, gate, word_segments, all_words_accepted))
    if not explanations:
        explanations.append("no strong retained or non-retained evidence found")

    return TokenScore(
        token=record,
        script_profile=profile,
        protected=protected,
        status=status,
        symbol_presets=sorted(set(presets)),
        dictionary_matches=dict_matches[:10],
        wordfreq_matches=wf_matches[:10],
        explanations=explanations,
    )


def decision_tree_status(
    *,
    protected: bool,
    gate_blocked: bool,
    has_word_segments: bool,
    all_words_accepted: bool,
) -> str:
    if protected:
        return "protected"
    if gate_blocked:
        return "non_retain_candidate"
    if not has_word_segments:
        return "retain_candidate"
    if all_words_accepted:
        return "retain_candidate"
    return "non_retain_candidate"


def _decision_explanation(
    status: str,
    protected: bool,
    gate: CharacterGate,
    word_segments: list[TokenSegment],
    all_words_accepted: bool,
) -> str:
    if protected:
        return "decision tree: accept protected token"
    if gate.blocked:
        return "decision tree: prune because the character gate failed"
    if not word_segments:
        return "decision tree: accept allowed symbol, escape, or short-fragment token"
    if all_words_accepted:
        return "decision tree: accept because every word segment was accepted"
    return "decision tree: prune because at least one word segment was not accepted"


@dataclass(frozen=True)
class CharacterGate:
    blocked: tuple[str, ...] = ()


def character_gate(
    text: str,
    config: ScoreConfig,
    segments: list[TokenSegment] | None = None,
) -> CharacterGate:
    blocked: set[str] = set()
    for segment in segments if segments is not None else token_segments(text):
        if segment.escaped:
            continue
        for char in segment.text:
            if char.isspace() or is_control_char(char):
                continue
            script = script_for_char(char)
            if char_matches_symbol_preset(char, config.symbol_presets):
                continue
            if script == "Common" and unicodedata.category(char)[0] in {"P", "S"}:
                blocked.add("Common-symbol")
                continue
            if script not in config.keep_scripts:
                blocked.add(script)
    return CharacterGate(tuple(sorted(blocked)))


def is_structural_token(text: str) -> bool:
    if not text:
        return False
    return all(char.isspace() or is_control_char(char) for char in text)


def is_control_char(char: str) -> bool:
    return unicodedata.category(char).startswith("C")


@dataclass(frozen=True)
class TokenSegment:
    text: str
    kind: str
    escaped: bool = False


def token_segments(text: str) -> list[TokenSegment]:
    segments: list[TokenSegment] = []
    index = 0
    while index < len(text):
        escape = ESCAPE_RE.match(text, index)
        if escape:
            segments.append(TokenSegment(escape.group(0), "escape", escaped=True))
            index = escape.end()
            continue
        char = text[index]
        kind = "word" if is_word_char(char) else "symbol"
        start = index
        index += 1
        while index < len(text):
            if ESCAPE_RE.match(text, index):
                break
            if ("word" if is_word_char(text[index]) else "symbol") != kind:
                break
            index += 1
        segments.append(TokenSegment(text[start:index], kind))
    return segments


def language_word_segments(segments: list[TokenSegment]) -> list[TokenSegment]:
    words = [segment for segment in segments if segment.kind == "word" and not segment.escaped]
    if all(len(segment.text) == 1 for segment in words):
        return []
    return [segment for segment in words if len(segment.text) > 1]


def language_evidence_segments(
    segments: list[TokenSegment],
    dictionaries: dict[str, set[str]],
) -> list[TokenSegment]:
    expanded: list[TokenSegment] = []
    terms = _dictionary_split_terms(dictionaries)
    for segment in segments:
        greedy_parts = _greedy_split(segment.text, terms)
        if greedy_parts:
            expanded.extend(TokenSegment(word, "word") for word in greedy_parts)
            continue

        parts = split_identifier_words(segment.text)
        for part in parts:
            greedy_parts = _greedy_split(part, terms)
            if greedy_parts:
                expanded.extend(TokenSegment(word, "word") for word in greedy_parts)
            elif len(part) > 1:
                expanded.append(TokenSegment(part, "word"))
    return expanded


def split_identifier_words(text: str) -> list[str]:
    parts = IDENTIFIER_PART_RE.findall(text)
    if not parts or "".join(parts) != text:
        return [text]
    return parts


def greedy_dictionary_split(text: str, dictionaries: dict[str, set[str]]) -> list[str]:
    return _greedy_split(text, _dictionary_split_terms(dictionaries))


def _dictionary_split_terms(dictionaries: dict[str, set[str]]) -> list[str]:
    return sorted(
        {
            term.lower()
            for dictionary in dictionaries.values()
            for term in dictionary
            if len(term) > 1 and term.isalpha()
        },
        key=lambda term: (-len(term), term),
    )


def _greedy_split(text: str, terms: list[str]) -> list[str]:
    normalized = text.lower()
    parts: list[str] = []
    index = 0
    while index < len(normalized):
        match = next((term for term in terms if normalized.startswith(term, index)), None)
        if match is None:
            return []
        parts.append(match)
        index += len(match)
    return parts if len(parts) > 1 else []


def is_word_char(char: str) -> bool:
    return unicodedata.category(char)[0] in {"L", "M"}


def has_language_evidence(text: str, dict_matches, wf_matches) -> bool:
    normalized = text.lower()
    return any(match.term == normalized or match.mode in {"exact", "subword"} for match in dict_matches + wf_matches)
