from tokenpruner.wordfreq_support import wordfreq_matches


def test_wordfreq_min_zipf_filters_low_frequency_terms():
    assert wordfreq_matches("hello", ["en"], min_zipf=8.0) == []


def test_wordfreq_min_zipf_allows_common_terms():
    matches = wordfreq_matches("hello", ["en"], min_zipf=4.0)

    assert matches
    assert matches[0].term == "hello"


def test_wordfreq_top_pct_filters_by_ranked_wordlist():
    assert wordfreq_matches("hello", ["en"], min_zipf=0.0, top_pct=0.001) == []
    assert wordfreq_matches("the", ["en"], min_zipf=0.0, top_pct=0.001)
