from vocabpruner.analysis.scoring import score_token, ScoreConfig
from vocabpruner.models import TokenRecord
import vocabpruner.analysis.wordfreq as analysis_wordfreq


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


def test_special_and_byte_fallback_are_protected():
    special = TokenRecord(0, "<|endoftext|>", "<|endoftext|>", is_special=True)
    byte = TokenRecord(1, "<0x20>", " ", is_byte_fallback=True)
    config = ScoreConfig(languages=())

    assert score_token(special, config).status == "protected"
    assert score_token(byte, config).status == "protected"


def test_script_preselection_prunes_non_retained_script():
    record = TokenRecord(2, "Ж", "Ж")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, languages=())

    score = score_token(record, config)

    assert score.status == "non_retain_candidate"


def test_disallowed_script_skips_wordfreq_and_dominates_punctuation():
    record = TokenRecord(4, "中文,", "中文,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"json"},
        symbol_lists={"ascii-punctuation"},
        languages=("en",),
    )

    score = score_token(record, config)

    assert score.status == "non_retain_candidate"
    assert any("character gate blocked" in explanation for explanation in score.explanations)


def test_unknown_latin_word_prunes_without_language_evidence():
    record = TokenRecord(9, "qzxq,", "qzxq,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_lists={"ascii-punctuation"},
        languages=("en",),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "non_retain_candidate"
    assert any("lack language evidence" in explanation for explanation in score.explanations)


def test_unknown_latin_word_does_not_match_json_literal_letters():
    record = TokenRecord(21, "Ġkeindahan", " keindahan")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"json"},
        languages=("en",),
    )

    score = score_token(record, config, dictionaries={})

    assert "json" not in score.syntax_lists
    assert not any("syntax lists matched: json" in explanation for explanation in score.explanations)
    assert score.status == "non_retain_candidate"


def test_selected_json_literal_is_retained_without_wordfreq():
    record = TokenRecord(22, "true", "true")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"json"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"
    assert score.syntax_lists == ["json"]


def test_known_word_plus_symbol_retains_with_wordfreq_evidence(monkeypatch):
    fake_wordfreq(monkeypatch)

    record = TokenRecord(10, "hello,", "hello,")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_lists={"ascii-punctuation"},
        languages=("en",),
        wordfreq_min_zipf=4.0,
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"
    assert score.wordfreq_matches


def test_symbol_only_token_retains_with_selected_list():
    record = TokenRecord(11, "=>", "=>")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"coding"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_greek_math_letters_retain_with_mathematics_list():
    record = TokenRecord(16, "λ", "λ")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_lists={"mathematics"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_greek_text_still_prunes_without_greek_script():
    record = TokenRecord(17, "λόγος", "λόγος")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_lists={"mathematics"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "non_retain_candidate"


def test_single_letters_separated_by_symbols_do_not_need_language_evidence():
    record = TokenRecord(12, "a+b", "a+b")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"coding"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_multi_letter_segments_separated_by_symbols_need_language_evidence():
    record = TokenRecord(13, "ab+cd", "ab+cd")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"coding"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "non_retain_candidate"
    assert any("ab" in explanation and "cd" in explanation for explanation in score.explanations)


def test_escaped_sequences_do_not_require_language_evidence():
    record = TokenRecord(14, "\\n", "\\n")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_lists=set(),
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"


def test_unicode_escape_sequence_is_structural_without_symbol_list():
    record = TokenRecord(18, "\\u1234", "\\u1234")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        symbol_lists=set(),
        languages=(),
    )

    score = score_token(record, config, dictionaries={})

    assert score.status == "retain_candidate"
    assert any("escape sequence" in explanation for explanation in score.explanations)


def test_structural_whitespace_is_protected():
    record = TokenRecord(5, "ĉĉĠ", "\t\t ")
    config = ScoreConfig(languages=())

    score = score_token(record, config)

    assert score.protected is True
    assert score.status == "protected"


def test_emoji_requires_selected_symbol_list():
    record = TokenRecord(6, "🙂", "🙂")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, symbol_lists=set(), languages=())

    score = score_token(record, config)

    assert score.status == "non_retain_candidate"


def test_dictionary_subword_match_counts_for_hello():
    record = TokenRecord(3, "he", "he")
    config = ScoreConfig(languages=())

    score = score_token(record, config, dictionaries={"english": {"hello"}})

    assert any(match.term == "hello" and match.mode == "subword" for match in score.dictionary_matches)


def test_camel_case_identifier_parts_count_as_language_evidence():
    record = TokenRecord(19, "classCallCheck", "classCallCheck")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"json"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={"programming": {"class", "call", "check"}})

    assert score.status == "retain_candidate"
    assert any(match.term == "class" and match.mode == "exact" for match in score.dictionary_matches)
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_joined_dictionary_words_count_as_language_evidence():
    record = TokenRecord(22, "typescript", "typescript")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        languages=(),
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
        languages=(),
    )

    score = score_token(record, config, dictionaries={"test": {"ab", "abc", "cd", "ef", "gh"}})

    assert score.status == "retain_candidate"
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_capitalization_boundaries_can_rescue_a_failed_greedy_split():
    record = TokenRecord(24, "abCd", "abCd")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={"test": {"ab", "abc", "cd"}})

    assert score.status == "retain_candidate"
    assert not any("lack language evidence" in explanation for explanation in score.explanations)


def test_partial_dictionary_evidence_does_not_override_tree_pruning():
    record = TokenRecord(20, "theros", "theros")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        syntax_lists={"json"},
        languages=(),
    )

    score = score_token(record, config, dictionaries={"english": {"the", "her"}})

    assert score.status == "non_retain_candidate"
    assert any("decision tree: prune" in explanation for explanation in score.explanations)


def test_wordfreq_default_zipf_threshold_filters_weaker_matches(monkeypatch):
    fake_wordfreq(monkeypatch)

    record = TokenRecord(7, "qzxq", "qzxq")
    config = ScoreConfig(keep_scripts={"Latin", "Common", "Inherited"}, languages=("en",))

    score = score_token(record, config, dictionaries={})

    assert score.wordfreq_matches == []


def test_wordfreq_lower_zipf_threshold_allows_weaker_matches(monkeypatch):
    fake_wordfreq(monkeypatch)

    record = TokenRecord(8, "congr", "congr")
    config = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        languages=("en",),
        wordfreq_min_zipf=1.0,
    )

    score = score_token(record, config, dictionaries={})

    assert score.wordfreq_matches


def test_wordfreq_threshold_changes_known_word_retention(monkeypatch):
    fake_wordfreq(monkeypatch)

    record = TokenRecord(15, "hello", "hello")
    strict = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        languages=("en",),
        wordfreq_min_zipf=8.0,
    )
    loose = ScoreConfig(
        keep_scripts={"Latin", "Common", "Inherited"},
        languages=("en",),
        wordfreq_min_zipf=4.0,
    )

    strict_score = score_token(record, strict, dictionaries={})
    loose_score = score_token(record, loose, dictionaries={})

    assert strict_score.status == "non_retain_candidate"
    assert loose_score.status == "retain_candidate"
