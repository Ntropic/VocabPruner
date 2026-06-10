from tokenpruner.models import TokenRecord
from tokenpruner.sampling import sample_records, score_sample
from tokenpruner.scoring import ScoreConfig, greedy_dictionary_split, score_token


def test_special_and_byte_fallback_are_protected():
    special = TokenRecord(0, "<|endoftext|>", "<|endoftext|>", is_special=True)
    byte = TokenRecord(1, "<0x20>", " ", is_byte_fallback=True)
    config = ScoreConfig(wordfreq_languages=())

    assert score_token(special, config).status == "protected"
    assert score_token(byte, config).status == "protected"


def test_script_preselection_prunes_non_retained_script():
    record = TokenRecord(2, "Ж", "Ж")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, wordfreq_languages=())

    score = score_token(record, config)

    assert score.status == "non_retain_candidate"


def test_disallowed_script_skips_wordfreq_and_dominates_punctuation():
    record = TokenRecord(4, "中文,", "中文,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"ascii-punctuation", "json"},
        wordfreq_languages=("en",),
    )

    score = score_token(record, config)

    assert score.status == "non_retain_candidate"
    assert any("character gate blocked" in explanation for explanation in score.explanations)


def test_unknown_latin_word_prunes_without_language_evidence():
    record = TokenRecord(9, "qzxq,", "qzxq,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"ascii-punctuation"},
        wordfreq_languages=("en",),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "non_retain_candidate"
    assert any("lack language evidence" in explanation for explanation in score.explanations)


def test_unknown_latin_word_does_not_match_json_literal_letters():
    record = TokenRecord(21, "Ġkeindahan", " keindahan")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"json"},
        wordfreq_languages=("en",),
    )

    score = score_token(record, config, dictionaries={})

    assert "json" not in score.symbol_presets
    assert not any("symbol/icon presets matched: json" in explanation for explanation in score.explanations)
    assert score.status == "non_retain_candidate"


def test_known_word_plus_symbol_retains_with_wordfreq_evidence():
    record = TokenRecord(10, "hello,", "hello,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"ascii-punctuation"},
        wordfreq_languages=("en",),
        wordfreq_min_zipf=4.0,
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"
    assert score.wordfreq_matches


def test_symbol_only_token_retains_with_selected_preset():
    record = TokenRecord(11, "=>", "=>")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"coding"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_greek_math_letters_retain_with_mathematics_preset():
    record = TokenRecord(16, "λ", "λ")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"mathematics"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_greek_text_still_prunes_without_greek_script():
    record = TokenRecord(17, "λόγος", "λόγος")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"mathematics"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "non_retain_candidate"


def test_single_letters_separated_by_symbols_do_not_need_language_evidence():
    record = TokenRecord(12, "a+b", "a+b")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"coding"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_multi_letter_segments_separated_by_symbols_need_language_evidence():
    record = TokenRecord(13, "ab+cd", "ab+cd")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"coding"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "non_retain_candidate"
    assert any("ab" in explanation and "cd" in explanation for explanation in score.explanations)


def test_escaped_sequences_do_not_require_language_evidence():
    record = TokenRecord(14, "\\n", "\\n")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets=set(),
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_unicode_escape_sequence_is_structural_without_symbol_preset():
    record = TokenRecord(18, "\\u1234", "\\u1234")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets=set(),
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"
    assert any("escape sequence" in explanation for explanation in score.explanations)


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

    assert score.status == "non_retain_candidate"


def test_dictionary_subword_match_counts_for_hello():
    record = TokenRecord(3, "he", "he")
    config = ScoreConfig(wordfreq_languages=())

    score = score_token(record, config, dictionaries={"english": {"hello"}})

    assert any(match.term == "hello" and match.mode == "subword" for match in score.dictionary_matches)


def test_camel_case_identifier_parts_count_as_language_evidence():
    record = TokenRecord(19, "classCallCheck", "classCallCheck")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"json"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={"programming": {"class", "call", "check"}})

    assert score.status == "retain_candidate"
    assert any(match.term == "class" and match.mode == "exact" for match in score.dictionary_matches)
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_greedy_dictionary_split_prefers_longest_word_at_each_position():
    dictionaries = {"test": {"ab", "abc", "cd", "de"}}

    assert greedy_dictionary_split("abcde", dictionaries) == ["abc", "de"]


def test_greedy_dictionary_split_requires_complete_coverage():
    dictionaries = {"english": {"the", "her"}}

    assert greedy_dictionary_split("theros", dictionaries) == []


def test_greedy_dictionary_split_does_not_backtrack():
    dictionaries = {"test": {"ab", "abc", "cd"}}

    assert greedy_dictionary_split("abcd", dictionaries) == []


def test_joined_dictionary_words_count_as_language_evidence():
    record = TokenRecord(22, "typescript", "typescript")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={"programming": {"type", "script"}})

    assert score.status == "retain_candidate"
    assert any(match.term == "type" and match.mode == "exact" for match in score.dictionary_matches)
    assert any(match.term == "script" and match.mode == "exact" for match in score.dictionary_matches)
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_capitalization_then_greedy_splitting_can_be_combined():
    record = TokenRecord(23, "abCdEfgh", "abCdEfgh")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={"test": {"ab", "abc", "cd", "ef", "gh"}})

    assert score.status == "retain_candidate"
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_capitalization_boundaries_can_rescue_a_failed_greedy_split():
    record = TokenRecord(24, "abCd", "abCd")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={"test": {"ab", "abc", "cd"}})

    assert score.status == "retain_candidate"
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_partial_dictionary_evidence_does_not_override_tree_pruning():
    record = TokenRecord(20, "theros", "theros")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_presets={"json"},
        wordfreq_languages=(),
    )

    score = score_token(record, config, dictionaries={"english": {"the", "her"}})

    assert score.status == "non_retain_candidate"
    assert any("decision tree: prune" in explanation for explanation in score.explanations)


def test_wordfreq_default_zipf_threshold_filters_weaker_matches():
    record = TokenRecord(7, "qzxq", "qzxq")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, wordfreq_languages=("en",))

    score = score_token(record, config, dictionaries={})

    assert score.wordfreq_matches == []


def test_wordfreq_lower_zipf_threshold_allows_weaker_matches():
    record = TokenRecord(8, "congr", "congr")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=("en",),
        wordfreq_min_zipf=1.0,
    )

    score = score_token(record, config, dictionaries={})

    assert score.wordfreq_matches


def test_wordfreq_threshold_changes_known_word_retention():
    record = TokenRecord(15, "hello", "hello")
    strict = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=("en",),
        wordfreq_min_zipf=8.0,
    )
    loose = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        wordfreq_languages=("en",),
        wordfreq_min_zipf=4.0,
    )

    strict_score = score_token(record, strict, dictionaries={})
    loose_score = score_token(record, loose, dictionaries={})

    assert strict_score.status == "non_retain_candidate"
    assert loose_score.status == "retain_candidate"


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
