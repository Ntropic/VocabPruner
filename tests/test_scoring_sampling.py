from tokenpruner.models import TokenRecord
from tokenpruner.sampling import sample_records, score_sample
from tokenpruner.scoring import ScoreConfig, score_token


def test_special_and_byte_fallback_are_protected():
    special = TokenRecord(0, "<|endoftext|>", "<|endoftext|>", is_special=True)
    byte = TokenRecord(1, "<0x20>", " ", is_byte_fallback=True)
    config = ScoreConfig(wordfreq_languages=())

    assert score_token(special, config).status == "protected"
    assert score_token(byte, config).status == "protected"


def test_script_preselection_scores_non_retained_script():
    record = TokenRecord(2, "Ж", "Ж")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, drop_scripts={"Cyrillic"}, wordfreq_languages=())

    score = score_token(record, config)

    assert score.non_retain_score > score.retain_score
    assert score.status == "non_retain_candidate"


def test_disallowed_script_skips_wordfreq_and_dominates_punctuation():
    record = TokenRecord(4, "中文,", "中文,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"ascii-punctuation", "json"},
        wordfreq_languages=("en",),
    )

    score = score_token(record, config)

    assert score.non_retain_score >= 100
    assert score.retain_score == 0
    assert score.wordfreq_score == 0
    assert score.status == "non_retain_candidate"
    assert any("character gate blocked" in explanation for explanation in score.explanations)


def test_structural_whitespace_is_protected():
    record = TokenRecord(5, "ĉĉĠ", "\t\t ")
    config = ScoreConfig(wordfreq_languages=())

    score = score_token(record, config)

    assert score.protected is True
    assert score.status == "protected"


def test_emoji_requires_selected_symbol_preset():
    record = TokenRecord(6, "🙂", "🙂")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, symbol_presets=set(), wordfreq_languages=())

    score = score_token(record, config)

    assert score.non_retain_score >= 100
    assert score.retain_score == 0


def test_dictionary_subword_score_counts_for_hello():
    record = TokenRecord(3, "he", "he")
    config = ScoreConfig(wordfreq_languages=())

    score = score_token(record, config, dictionaries={"english": {"hello"}})

    assert score.dictionary_score > 0
    assert any(match.term == "hello" and match.mode == "subword" for match in score.dictionary_matches)


def test_wordfreq_default_zipf_threshold_filters_weaker_matches():
    record = TokenRecord(7, "qzxq", "qzxq")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, wordfreq_languages=("en",))

    score = score_token(record, config, dictionaries={})

    assert score.wordfreq_score == 0
    assert score.wordfreq_matches == []


def test_wordfreq_lower_zipf_threshold_allows_weaker_matches():
    record = TokenRecord(8, "congr", "congr")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=("en",),
        wordfreq_min_zipf=1.0,
    )

    score = score_token(record, config, dictionaries={})

    assert score.wordfreq_score > 0


def test_sampling_is_deterministic():
    records = [TokenRecord(i, str(i), str(i)) for i in range(20)]

    first = [record.token_id for record in sample_records(records, 5, 7)]
    second = [record.token_id for record in sample_records(records, 5, 7)]

    assert first == second


def test_score_sample_only_scores_sample_size():
    records = [TokenRecord(i, "hello", "hello") for i in range(20)]
    scores = score_sample(records, ScoreConfig(wordfreq_languages=()), 4, 1)

    assert len(scores) == 4


def test_score_sample_minus_one_scores_all_records():
    records = [TokenRecord(i, "hello", "hello") for i in range(20)]
    scores = score_sample(records, ScoreConfig(wordfreq_languages=()), -1, 1)

    assert len(scores) == 20
    assert [score.token.token_id for score in scores] == list(range(20))


def test_process_score_sample_preserves_sample_order():
    records = [TokenRecord(i, f"hello{i}", f"hello{i}") for i in range(20)]
    config = ScoreConfig(wordfreq_languages=())

    serial = score_sample(records, config, 8, 3, workers=1)
    parallel = score_sample(records, config, 8, 3, workers=2)

    assert [score.token.token_id for score in parallel] == [score.token.token_id for score in serial]
