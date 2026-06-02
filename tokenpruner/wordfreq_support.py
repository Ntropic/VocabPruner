from __future__ import annotations

from functools import lru_cache

from tokenpruner.dictionaries import normalize_token_text
from tokenpruner.models import WordfreqMatch


def wordfreq_matches(
    text: str,
    languages: list[str] | None,
    min_zipf: float = 4.0,
    top_pct: float | None = None,
) -> list[WordfreqMatch]:
    if not languages:
        return []
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return []

    token = normalize_token_text(text)
    if not token:
        return []

    matches: list[WordfreqMatch] = []
    for language in languages:
        frequency = zipf_frequency(token, language)
        if _passes_wordfreq_threshold(language, token, frequency, min_zipf, top_pct):
            matches.append(_match(language, token, token, "exact", frequency))
            continue

        for word in _wordlist(language):
            word = word.lower()
            mode = _wordfreq_match_mode(token, word)
            if mode is None:
                continue
            word_frequency = zipf_frequency(word, language)
            if not _passes_wordfreq_threshold(language, word, word_frequency, min_zipf, top_pct):
                continue
            matches.append(_match(language, token, word, mode, word_frequency))
            if len(matches) >= 20:
                break
    return sorted(matches, key=lambda match: match.score, reverse=True)


def _match(language: str, token: str, term: str, mode: str, frequency: float) -> WordfreqMatch:
    weight = {"exact": 1.2, "subword": 0.55, "prefix": 0.75, "suffix": 0.65}[mode]
    return WordfreqMatch(
        language=language,
        term=term,
        mode=mode,
        frequency=frequency,
        score=max(0.0, min(2.0, frequency / 4.0 * weight)),
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
) -> bool:
    if frequency < min_zipf:
        return False
    if top_pct is None:
        return True
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
