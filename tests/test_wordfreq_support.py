import vocabpruner.wordfreq_support as support
from vocabpruner.wordfreq_support import wordfreq_matches


def setup_module():
    support._wordlist.cache_clear()
    support._word_rank_index.cache_clear()


def setup_function():
    support._wordlist.cache_clear()
    support._word_rank_index.cache_clear()


def fake_wordfreq(monkeypatch):
    monkeypatch.setattr(support, "_wordlist", lambda language: ("the", "hello", "congratulations", "congruent"))
    monkeypatch.setattr(
        support,
        "_zipf_frequency",
        lambda word, language: {
            "the": 7.7,
            "hello": 4.7,
            "congratulations": 3.8,
            "congruent": 3.0,
        }.get(word.lower(), 0.0),
    )
    monkeypatch.setattr(
        support,
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

    assert wordfreq_matches("hello", ["en"], min_zipf=8.0) == []


def test_wordfreq_min_zipf_allows_common_terms(monkeypatch):
    fake_wordfreq(monkeypatch)

    matches = wordfreq_matches("hello", ["en"], min_zipf=4.0)

    assert matches
    assert matches[0].term == "hello"


def test_wordfreq_top_pct_filters_by_ranked_wordlist(monkeypatch):
    fake_wordfreq(monkeypatch)

    assert wordfreq_matches("hello", ["en"], min_zipf=0.0, top_pct=0.001) == []
    assert wordfreq_matches("the", ["en"], min_zipf=0.0, top_pct=0.001)
