from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re
import unicodedata

import numpy as np

from vocabpruner.analysis.dictionaries import dictionary_matches, load_dictionaries
from vocabpruner.analysis.scripts import normalize_scripts, script_profile, scripts_for_char
from vocabpruner.analysis.symbols import (
    char_matches_list,
    matching_symbol_lists,
    matching_syntax_lists,
    matching_syntax_word_lists,
    normalize_symbol_lists,
    normalize_syntax_lists,
)
from vocabpruner.analysis.wordfreq import wordfreq_diagnostics
from vocabpruner.models import (
    DictionaryMatch,
    ScriptProfile,
    TokenRecord,
    TokenScore,
    WordfreqMatch,
)


ESCAPE_RE = re.compile(r"""\\(?:[nrtbfav\\"']|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})""")


IDENTIFIER_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+")


@dataclass(frozen=True)
class ScoreConfig:
    keep_scripts: set[str] = field(default_factory=lambda: {"Latin", "Common", "Inherited"})
    syntax_lists: set[str] = field(default_factory=set)
    symbol_lists: set[str] = field(default_factory=set)
    languages: tuple[str, ...] = ("en",)
    wordfreq_min_zipf: float = 4.0
    wordfreq_top_pct: float | None = None
    no_wordfreq: bool = False
    custom_dictionaries: tuple[str, ...] = ()

    @classmethod
    def from_cli(
        cls,
        keep_scripts: list[str] | None = None,
        syntax_lists: list[str] | None = None,
        symbol_lists: list[str] | None = None,
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
            syntax_lists=normalize_syntax_lists(syntax_lists or []),
            symbol_lists=normalize_symbol_lists(symbol_lists or []),
            languages=tuple(languages or ["en"]),
            wordfreq_min_zipf=4.0 if wordfreq_min_zipf is None and wordfreq_top_pct is None else (wordfreq_min_zipf or 0.0),
            wordfreq_top_pct=wordfreq_top_pct,
            no_wordfreq=wordfreq_min_zipf is None and wordfreq_top_pct is None,
            custom_dictionaries=tuple(dictionaries or []),
        )


@dataclass(frozen=True)
class PipelineArrays:
    token_ids: np.ndarray
    retain_candidates: np.ndarray
    protected: np.ndarray
    wordfreq_token_ids: np.ndarray
    wordfreq_segment_indexes: np.ndarray
    language_ids: np.ndarray
    zipf: np.ndarray
    rank_percent: np.ndarray
    wordfreq_passes: np.ndarray
    best_zipf_by_segment: dict[tuple[int, int], float]
    best_rank_percent_by_segment: dict[tuple[int, int], float]
    segment_wordfreq_passes: dict[tuple[int, int], bool]


def settings_config(
    keep_script: list[str],
    syntax_list: list[str],
    symbol_list: list[str],
    dictionary: list[str],
    languages: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
) -> ScoreConfig:
    try:
        return ScoreConfig.from_cli(
            keep_scripts=keep_script,
            syntax_lists=syntax_list,
            symbol_lists=symbol_list,
            languages=languages,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
            dictionaries=dictionary,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def score_token(record: TokenRecord, config: ScoreConfig, dictionaries: dict[str, set[str]] | None = None) -> TokenScore:
    dictionaries = dictionaries if dictionaries is not None else load_dictionaries(list(config.custom_dictionaries))
    _, fact, rows = token_fact_row(record, config, dictionaries)
    token_facts = {"columns": {key: [value] for key, value in fact.items()}}
    wordfreq_facts = {"columns": {
        key: [row[key] for row in rows]
        for key in ("token_id", "segment_index", "language", "term", "mode", "zipf", "rank", "rank_percent")
    }}
    arrays = arrays_from_facts(token_facts, wordfreq_facts, config)
    return derive_scores([record], token_facts, wordfreq_facts, config, arrays)[0]


def deterministic_token_fact(record: TokenRecord, config: ScoreConfig, dictionaries: dict[str, set[str]]) -> dict[str, Any]:
    text = record.decoded_text or record.raw_token
    profile = script_profile(text)
    structural = is_structural_token(text)
    protected = record.is_special or record.is_byte_fallback or structural
    segments = token_segments(text)
    gate = character_gate(text, config, segments)
    syntax_matches = matching_syntax_lists(text, config.syntax_lists)
    symbol_matches = matching_symbol_lists(text, config.symbol_lists)
    word_segments = language_word_segments(segments)
    evidence_segments = language_evidence_segments(word_segments, dictionaries)
    dict_matches_by_segment = [dictionary_matches(segment.text, dictionaries) for segment in evidence_segments]
    dictionary_accepted = [
        bool(matching_syntax_word_lists(segment.text, config.syntax_lists))
        or has_language_evidence(segment.text, segment_matches, [])
        for segment, segment_matches in zip(evidence_segments, dict_matches_by_segment)
    ]
    explanations: list[str] = []
    if record.is_special:
        explanations.append("special token is always preserved")
    if record.is_byte_fallback:
        explanations.append("byte fallback token is always preserved")
    if structural:
        explanations.append("decoded whitespace/control token is always preserved")
    if not protected and gate.blocked:
        explanations.append(f"character gate blocked disallowed character families: {', '.join(gate.blocked)}")
    elif not protected:
        explanations.append("character gate passed")
    if syntax_matches and (protected or not gate.blocked):
        explanations.append(f"syntax lists matched: {', '.join(syntax_matches)}")
    if symbol_matches and (protected or not gate.blocked):
        explanations.append(f"symbol lists matched: {', '.join(symbol_matches)}")
    flat_dict_matches = [match.to_dict() for matches in dict_matches_by_segment for match in matches[:10]]
    if flat_dict_matches:
        top = ", ".join(f"{match['dictionary']}:{match['term']}/{match['mode']}" for match in flat_dict_matches[:3])
        explanations.append(f"dictionary evidence matched: {top}")
    if not protected and not gate.blocked and not word_segments and (syntax_matches or symbol_matches):
        explanations.append("token retained by selected syntax or symbol lists")
    elif not protected and not gate.blocked and not word_segments and any(segment.escaped for segment in segments):
        explanations.append("escape sequence retained as structural syntax")
    retain_candidate = protected or not gate.blocked
    return {
        "token_id": record.token_id,
        "raw_token": record.raw_token,
        "decoded_text": record.decoded_text,
        "is_special": record.is_special,
        "is_byte_fallback": record.is_byte_fallback,
        "script_counts": dict(profile.counts),
        "total_meaningful": profile.total_meaningful,
        "low_confidence": profile.low_confidence,
        "protected": protected,
        "gate_blocked": list(gate.blocked),
        "retain_candidate": retain_candidate,
        "has_word_segments": bool(word_segments),
        "evidence_segments": [segment.text for segment in evidence_segments],
        "dictionary_accepted": dictionary_accepted,
        "syntax_lists": syntax_matches,
        "symbol_lists": symbol_matches,
        "dictionary_matches": flat_dict_matches[:10],
        "explanations": explanations,
    }


def arrays_from_facts(token_facts: dict[str, Any], wordfreq_facts: dict[str, Any], config: ScoreConfig) -> PipelineArrays:
    token_columns = token_facts["columns"]
    wordfreq_columns = wordfreq_facts["columns"]
    language_to_id = {language: index for index, language in enumerate(config.languages)}
    rank_percent = [value if value is not None else np.inf for value in wordfreq_columns["rank_percent"]]
    zipf = np.asarray(wordfreq_columns["zipf"], dtype=np.float64)
    rank_percent_array = np.asarray(rank_percent, dtype=np.float64)
    if config.wordfreq_top_pct is not None:
        wordfreq_passes = rank_percent_array <= config.wordfreq_top_pct
    else:
        wordfreq_passes = zipf >= config.wordfreq_min_zipf
    token_ids = np.asarray(wordfreq_columns["token_id"], dtype=np.int64)
    segment_indexes = np.asarray(wordfreq_columns["segment_index"], dtype=np.int64)
    best_zipf = _best_by_segment(token_ids, segment_indexes, zipf, reducer=max)
    best_rank_percent = _best_by_segment(token_ids, segment_indexes, rank_percent_array, reducer=min)
    segment_passes = _segment_passes(token_ids, segment_indexes, wordfreq_passes)
    return PipelineArrays(
        token_ids=np.asarray(token_columns["token_id"], dtype=np.int64),
        retain_candidates=np.asarray(token_columns["retain_candidate"], dtype=bool),
        protected=np.asarray(token_columns["protected"], dtype=bool),
        wordfreq_token_ids=token_ids,
        wordfreq_segment_indexes=segment_indexes,
        language_ids=np.asarray([language_to_id.get(language, -1) for language in wordfreq_columns["language"]], dtype=np.int64),
        zipf=zipf,
        rank_percent=rank_percent_array,
        wordfreq_passes=wordfreq_passes,
        best_zipf_by_segment=best_zipf,
        best_rank_percent_by_segment=best_rank_percent,
        segment_wordfreq_passes=segment_passes,
    )


def _best_by_segment(token_ids: np.ndarray, segment_indexes: np.ndarray, values: np.ndarray, reducer) -> dict[tuple[int, int], float]:
    best: dict[tuple[int, int], float] = {}
    for token_id, segment_index, value in zip(token_ids, segment_indexes, values):
        if np.isinf(value):
            continue
        key = (int(token_id), int(segment_index))
        best[key] = float(value) if key not in best else float(reducer(best[key], float(value)))
    return best


def _segment_passes(token_ids: np.ndarray, segment_indexes: np.ndarray, passes: np.ndarray) -> dict[tuple[int, int], bool]:
    passed: dict[tuple[int, int], bool] = {}
    for token_id, segment_index, does_pass in zip(token_ids, segment_indexes, passes):
        key = (int(token_id), int(segment_index))
        passed[key] = passed.get(key, False) or bool(does_pass)
    return passed


def derive_scores(
    records: list[TokenRecord],
    token_facts: dict[str, Any],
    wordfreq_facts: dict[str, Any],
    config: ScoreConfig,
    arrays: PipelineArrays,
) -> list[TokenScore]:
    token_columns = token_facts["columns"]
    wordfreq_by_token_segment: dict[tuple[int, int], list[dict[str, Any]]] = {}
    wordfreq_columns = wordfreq_facts["columns"]
    for index, token_id in enumerate(wordfreq_columns["token_id"]):
        key = (token_id, wordfreq_columns["segment_index"][index])
        wordfreq_by_token_segment.setdefault(key, []).append({column: values[index] for column, values in wordfreq_columns.items()})
    records_by_id = {record.token_id: record for record in records}
    scores: list[TokenScore] = []
    for index, token_id in enumerate(token_columns["token_id"]):
        record = records_by_id.get(token_id) or TokenRecord(
            token_id=token_id,
            raw_token=token_columns["raw_token"][index],
            decoded_text=token_columns["decoded_text"][index],
            is_special=token_columns["is_special"][index],
            is_byte_fallback=token_columns["is_byte_fallback"][index],
        )
        protected = bool(token_columns["protected"][index])
        retain_candidate = bool(token_columns["retain_candidate"][index])
        evidence_segments = list(token_columns["evidence_segments"][index])
        dictionary_accepted = list(token_columns["dictionary_accepted"][index])
        explanations = list(token_columns["explanations"][index])
        passing_wordfreq: list[WordfreqMatch] = []
        segment_acceptance: list[bool] = []
        for segment_index, segment in enumerate(evidence_segments):
            accepted = bool(dictionary_accepted[segment_index])
            diagnostics = wordfreq_by_token_segment.get((token_id, segment_index), [])
            segment_wordfreq_passes = arrays.segment_wordfreq_passes.get((int(token_id), segment_index), False)
            if segment_wordfreq_passes:
                segment_matches = [_diagnostic_to_match(item) for item in diagnostics if _diagnostic_passes_threshold(item, config)]
                accepted = True
                passing_wordfreq.extend(segment_matches)
            segment_acceptance.append(accepted)
        if passing_wordfreq:
            passing_wordfreq.sort(key=lambda match: match.frequency, reverse=True)
            top = ", ".join(f"{match.language}:{match.term}/{match.mode}" for match in passing_wordfreq[:3])
            explanations.append(f"wordfreq evidence matched: {top}")

        if protected:
            status = "protected"
            explanations.append("decision tree: accept protected token")
        elif not retain_candidate:
            status = "non_retain_candidate"
            explanations.append("decision tree: prune because the character gate failed")
        elif config.no_wordfreq:
            status = "retain_candidate"
            explanations.append("decision tree: accept because wordfreq filtering is disabled")
        elif not evidence_segments:
            status = "retain_candidate"
            explanations.append("decision tree: accept allowed symbol, escape, or short-fragment token")
        elif all(segment_acceptance):
            status = "retain_candidate"
            explanations.append("decision tree: accept because every word segment was accepted")
        else:
            status = "non_retain_candidate"
            missing = [segment for segment, accepted in zip(evidence_segments, segment_acceptance) if not accepted]
            if missing:
                explanations.append(f"word segment(s) lack language evidence: {', '.join(missing[:5])}")
            explanations.append("decision tree: prune because at least one word segment was not accepted")

        scores.append(
            TokenScore(
                token=record,
                script_profile=ScriptProfile(
                    counts=dict(token_columns["script_counts"][index]),
                    total_meaningful=token_columns["total_meaningful"][index],
                    low_confidence=token_columns["low_confidence"][index],
                ),
                protected=protected,
                status=status,
                syntax_lists=list(token_columns["syntax_lists"][index]),
                symbol_lists=list(token_columns["symbol_lists"][index]),
                dictionary_matches=[DictionaryMatch(**match) for match in token_columns["dictionary_matches"][index]],
                wordfreq_matches=passing_wordfreq[:10],
                explanations=explanations,
            )
        )
    return sorted(scores, key=lambda score: score.token.token_id)


def _diagnostic_to_match(item: dict[str, Any]) -> WordfreqMatch:
    return WordfreqMatch(
        language=item["language"],
        term=item["term"],
        mode=item["mode"],
        frequency=item["zipf"],
        rank=item["rank"],
        rank_percent=item["rank_percent"],
    )


def _diagnostic_passes_threshold(item: dict[str, Any], config: ScoreConfig) -> bool:
    if config.wordfreq_top_pct is not None:
        rank_percent = item["rank_percent"]
        return rank_percent is not None and rank_percent <= config.wordfreq_top_pct
    return item["zipf"] >= config.wordfreq_min_zipf


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
            scripts = scripts_for_char(char)
            if char_matches_list(char, config.syntax_lists, config.symbol_lists):
                continue
            if "Common" in scripts and unicodedata.category(char)[0] in {"P", "S"}:
                blocked.add("Common-symbol")
                continue
            if not scripts & config.keep_scripts:
                blocked.update(scripts)
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


def token_fact_row(
    record: TokenRecord,
    config: ScoreConfig,
    dictionaries: dict[str, set[str]],
) -> tuple[TokenRecord, dict[str, Any], list[dict[str, Any]]]:
    fact = deterministic_token_fact(record, config, dictionaries)
    wordfreq_rows: list[dict[str, Any]] = []
    if fact["retain_candidate"] and not fact["protected"] and not config.no_wordfreq:
        for segment_index, segment in enumerate(fact["evidence_segments"]):
            for diagnostic in wordfreq_diagnostics(segment, list(config.languages)):
                wordfreq_rows.append(
                    {
                        "token_id": record.token_id,
                        "segment_index": segment_index,
                        "language": diagnostic.language,
                        "term": diagnostic.term,
                        "mode": diagnostic.mode,
                        "zipf": diagnostic.frequency,
                        "rank": diagnostic.rank,
                        "rank_percent": diagnostic.rank_percent,
                    }
                )
    return record, fact, wordfreq_rows
