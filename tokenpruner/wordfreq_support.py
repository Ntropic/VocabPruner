from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

from tokenpruner.dictionaries import normalize_token_text
from tokenpruner.models import WordfreqMatch

@dataclass(frozen=True)
class WordfreqDiagnostic:
    language: str
    term: str
    mode: str
    frequency: float
    rank: int | None
    rank_percent: float | None


def wordfreq_matches(
    text: str,
    languages: list[str] | None,
    min_zipf: float = 4.0,
    top_pct: float | None = None,
) -> list[WordfreqMatch]:
    """Compatibility wrapper over threshold-independent diagnostics."""
    if not languages:
        return []
    matches: list[WordfreqMatch] = []
    for diagnostic in wordfreq_diagnostics(text, languages):
        if _passes_wordfreq_threshold(
            diagnostic.language,
            diagnostic.term,
            diagnostic.frequency,
            min_zipf,
            top_pct,
            rank_percent=diagnostic.rank_percent,
        ):
            matches.append(
                _match(
                    diagnostic.language,
                    diagnostic.term,
                    diagnostic.mode,
                    diagnostic.frequency,
                    diagnostic.rank,
                    diagnostic.rank_percent,
                )
            )
    return sorted(matches, key=lambda match: match.frequency, reverse=True)


def wordfreq_diagnostics(
    text: str,
    languages: list[str] | None,
    limit: int = 20,
) -> list[WordfreqDiagnostic]:
    if not languages:
        return []
    token = normalize_token_text(text)
    if not token:
        return []

    diagnostics: list[WordfreqDiagnostic] = []
    for language in languages:
        frequency = _zipf_frequency(token, language)
        rank, rank_percent = _rank_info(language, token)
        if frequency > 0 or rank is not None:
            diagnostics.append(
                WordfreqDiagnostic(
                    language=language,
                    term=token,
                    mode="exact",
                    frequency=frequency,
                    rank=rank,
                    rank_percent=rank_percent,
                )
            )

        language_count = 1
        for word in _wordlist(language):
            word = word.lower()
            mode = _wordfreq_match_mode(token, word)
            if mode is None or mode == "exact":
                continue
            word_frequency = _zipf_frequency(word, language)
            word_rank, word_rank_percent = _rank_info(language, word)
            diagnostics.append(
                WordfreqDiagnostic(
                    language=language,
                    term=word,
                    mode=mode,
                    frequency=word_frequency,
                    rank=word_rank,
                    rank_percent=word_rank_percent,
                )
            )
            language_count += 1
            if language_count >= limit:
                break
    return sorted(diagnostics, key=lambda item: item.frequency, reverse=True)


def _match(
    language: str,
    term: str,
    mode: str,
    frequency: float,
    rank: int | None,
    rank_percent: float | None,
) -> WordfreqMatch:
    return WordfreqMatch(
        language=language,
        term=term,
        mode=mode,
        frequency=frequency,
        rank=rank,
        rank_percent=rank_percent,
    )


def _wordfreq_match_mode(token: str, word: str) -> str | None:
    if token == word:
        return "exact"
    if len(token) >= 2 and word.startswith(token):
        return "subword"
    if len(token) >= 3 and word.endswith(token):
        return "subword"
    if len(token) >= 3 and token in word:
        return "subword"
    return None


def _passes_wordfreq_threshold(
    language: str,
    word: str,
    frequency: float,
    min_zipf: float,
    top_pct: float | None,
    rank_percent: float | None = None,
) -> bool:
    if frequency < min_zipf:
        return False
    if top_pct is None:
        return True
    if rank_percent is not None:
        return rank_percent <= top_pct
    return _word_in_top_pct(language, word, top_pct)


def _word_in_top_pct(language: str, word: str, top_pct: float) -> bool:
    words = _wordlist(language)
    if not words:
        return False
    cutoff = max(1, int(len(words) * (top_pct / 100.0)))
    return word in set(words[:cutoff])


@lru_cache(maxsize=128)
def _wordlist(language: str) -> tuple[str, ...]:
    try:
        from wordfreq import iter_wordlist
    except ImportError:
        return ()
    return tuple(word.lower() for word in iter_wordlist(language, wordlist="small"))


def _zipf_frequency(word: str, language: str) -> float:
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return 0.0
    return zipf_frequency(word, language)


@lru_cache(maxsize=128)
def _word_rank_index(language: str) -> dict[str, int]:
    return {word: index + 1 for index, word in enumerate(_wordlist(language))}


def _rank_info(language: str, word: str) -> tuple[int | None, float | None]:
    words = _wordlist(language)
    if not words:
        return None, None
    rank = _word_rank_index(language).get(word.lower())
    if rank is None:
        return None, None
    return rank, rank / len(words) * 100.0
