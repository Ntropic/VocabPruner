from vocabpruner.analysis.scoring import score_token, ScoreConfig
from vocabpruner.models import TokenRecord
import vocabpruner.analysis.wordfreq as analysis_wordfreq


def setup_module():
    analysis_wordfreq._wordlist.cache_clear()
    analysis_wordfreq._word_rank_index.cache_clear()


def setup_function():
    analysis_wordfreq._wordlist.cache_clear()
    analysis_wordfreq._word_rank_index.cache_clear()


def fake_wordfreq(monkeypatch):
    monkeypatch.setattr(analysis_wordfreq, "_wordlist", lambda language: ("the", "hello", "congratulations", "congruent"))
    monkeypatch.setattr(
        analysis_wordfreq,
        "_zipf_frequency",
        lambda word, language: {
            "the": 7.7,
            "hello": 4.7,
            "congratulations": 3.8,
            "congruent": 3.0,
        }.get(word.lower(), 0.0),
    )
    monkeypatch.setattr(
        analysis_wordfreq,
        "_rank_info",
        lambda language, word: {
            "the": (1, 0.0005),
            "hello": (2, 0.01),
            "congratulations": (3, 0.02),
            "congruent": (4, 0.03),
        }.get(word.lower(), (None, None)),
    )


def test_wordfreq_min_zipf_filters_low_frequency_terms(monkeypatch):
    fake_wordfreq(monkeypatch)

    score = score_token(TokenRecord(0, "hello", "hello"), ScoreConfig(languages=("en",), wordfreq_min_zipf=8.0))
    assert score.wordfreq_matches == []


def test_wordfreq_min_zipf_allows_common_terms(monkeypatch):
    fake_wordfreq(monkeypatch)

    matches = score_token(TokenRecord(0, "hello", "hello"), ScoreConfig(languages=("en",), wordfreq_min_zipf=4.0)).wordfreq_matches

    assert matches
    assert matches[0].term == "hello"


def test_wordfreq_top_pct_filters_by_ranked_wordlist(monkeypatch):
    fake_wordfreq(monkeypatch)

    config = ScoreConfig(languages=("en",), wordfreq_min_zipf=0.0, wordfreq_top_pct=0.001)
    assert score_token(TokenRecord(0, "hello", "hello"), config).wordfreq_matches == []
    assert score_token(TokenRecord(1, "the", "the"), config).wordfreq_matches
