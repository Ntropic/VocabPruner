from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata

from tokenpruner.dictionaries import dictionary_matches, load_dictionaries
from tokenpruner.models import TokenRecord, TokenScore
from tokenpruner.scripts import normalize_scripts, script_for_char, script_profile
from tokenpruner.symbols import char_matches_symbol_preset, matching_symbol_presets, normalize_presets
from tokenpruner.wordfreq_support import wordfreq_matches


@dataclass(frozen=True)
class ScoreConfig:
    keep_scripts: set[str] = field(default_factory=lambda: {"Latin", "Common", "Inherited"})
    drop_scripts: set[str] = field(default_factory=set)
    symbol_presets: set[str] = field(default_factory=lambda: {"coding", "ascii-punctuation"})
    wordfreq_languages: tuple[str, ...] = ("en",)
    wordfreq_min_zipf: float = 4.0
    wordfreq_top_pct: float | None = None
    custom_dictionaries: tuple[str, ...] = ()
    builtin_dictionaries: tuple[str, ...] | None = None

    @classmethod
    def from_cli(
        cls,
        keep_scripts: list[str] | None = None,
        drop_scripts: list[str] | None = None,
        symbol_presets: list[str] | None = None,
        wordfreq_languages: list[str] | None = None,
        wordfreq_min_zipf: float = 4.0,
        wordfreq_top_pct: float | None = None,
        dictionaries: list[str] | None = None,
    ) -> "ScoreConfig":
        if wordfreq_min_zipf < 0:
            raise ValueError("wordfreq_min_zipf must be non-negative")
        if wordfreq_top_pct is not None and not 0 < wordfreq_top_pct <= 100:
            raise ValueError("wordfreq_top_pct must be greater than 0 and at most 100")
        return cls(
            keep_scripts=normalize_scripts(keep_scripts or ["Latin", "Common", "Inherited"]),
            drop_scripts=normalize_scripts(drop_scripts or []),
            symbol_presets=normalize_presets(symbol_presets or ["coding", "ascii-punctuation"]),
            wordfreq_languages=tuple(wordfreq_languages or ["en"]),
            wordfreq_min_zipf=wordfreq_min_zipf,
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

    retain_score = 0.0
    non_retain_score = 0.0
    explanations: list[str] = []

    structural = is_structural_token(text)
    protected = record.is_special or record.is_byte_fallback or structural
    if record.is_special:
        retain_score += 100.0
        explanations.append("special token is always preserved")
    if record.is_byte_fallback:
        retain_score += 100.0
        explanations.append("byte fallback token is always preserved")
    if structural:
        retain_score += 100.0
        explanations.append("decoded whitespace/control token is always preserved")

    presets = matching_symbol_presets(text, config.symbol_presets)
    gate = character_gate(text, config)

    if not protected and gate.blocked:
        non_retain_score += 100.0
        explanations.append(f"character gate blocked disallowed character families: {', '.join(gate.blocked)}")
    elif not protected:
        for script, count in profile.counts.items():
            ratio = count / max(1, profile.total_meaningful)
            strength = 1.0 if not profile.low_confidence else 0.55
            if script in config.keep_scripts:
                retain_score += 3.0 * ratio * strength
                explanations.append(f"{script} script contributes retained score {3.0 * ratio * strength:.2f}")

    symbol_score = 0.0
    if presets and (protected or not gate.blocked):
        symbol_score = 1.25 * len(set(presets))
        retain_score += symbol_score
        explanations.append(f"symbol/icon presets matched: {', '.join(sorted(set(presets)))}")

    dict_matches = []
    dictionary_score = 0.0
    wf_matches = []
    wordfreq_score = 0.0
    if not protected and not gate.blocked:
        dict_matches = dictionary_matches(text, dictionaries)
        dictionary_score = min(4.0, sum(match.score for match in dict_matches[:5]))
    if dictionary_score:
        retain_score += dictionary_score
        top = ", ".join(f"{match.dictionary}:{match.term}/{match.mode}" for match in dict_matches[:3])
        explanations.append(f"dictionary evidence matched: {top}")

    if not protected and not gate.blocked:
        wf_matches = wordfreq_matches(
            text,
            list(config.wordfreq_languages),
            min_zipf=config.wordfreq_min_zipf,
            top_pct=config.wordfreq_top_pct,
        )
        wordfreq_score = min(3.0, sum(match.score for match in wf_matches[:5]))
    if wordfreq_score:
        retain_score += wordfreq_score
        top = ", ".join(f"{match.language}:{match.term}/{match.mode}" for match in wf_matches[:3])
        explanations.append(f"wordfreq evidence matched: {top}")

    status = classify_status(retain_score, non_retain_score, protected)
    if not explanations:
        explanations.append("no strong retained or non-retained evidence found")

    return TokenScore(
        token=record,
        script_profile=profile,
        retain_score=retain_score,
        non_retain_score=non_retain_score,
        symbol_score=symbol_score,
        dictionary_score=dictionary_score,
        wordfreq_score=wordfreq_score,
        protected=protected,
        status=status,
        symbol_presets=sorted(set(presets)),
        dictionary_matches=dict_matches[:10],
        wordfreq_matches=wf_matches[:10],
        explanations=explanations,
    )


def classify_status(retain_score: float, non_retain_score: float, protected: bool) -> str:
    if protected:
        return "protected"
    if abs(retain_score - non_retain_score) <= 0.75:
        return "ambiguous"
    if retain_score > non_retain_score:
        return "retain_candidate"
    return "non_retain_candidate"


@dataclass(frozen=True)
class CharacterGate:
    blocked: tuple[str, ...] = ()


def character_gate(text: str, config: ScoreConfig) -> CharacterGate:
    blocked: set[str] = set()
    for char in text:
        if char.isspace() or is_control_char(char):
            continue
        script = script_for_char(char)
        if char_matches_symbol_preset(char, config.symbol_presets):
            continue
        if script == "Common" and unicodedata.category(char)[0] in {"P", "S"}:
            blocked.add("Common-symbol")
            continue
        if script in config.drop_scripts:
            blocked.add(script)
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
