import json
from importlib import resources

from vocabpruner.scoring import ScoreConfig, character_gate
from vocabpruner.scripts import normalize_scripts, script_lists, script_profile, scripts_for_char
from vocabpruner.symbols import (
    _load_list,
    matching_symbol_lists,
    matching_syntax_lists,
    matching_syntax_word_lists,
    normalize_symbol_lists,
    normalize_syntax_lists,
    symbol_lists,
    syntax_lists,
)


def test_script_profile_detects_supported_scripts():
    profile = script_profile("hello Ж中あ한")

    assert profile.counts["Latin"] == 5
    assert profile.counts["Cyrillic"] == 1
    assert profile.counts["Han"] == 1
    assert profile.counts["Hiragana"] == 1
    assert profile.counts["Hangul"] == 1


def test_short_fragments_are_low_confidence():
    assert script_profile("he").low_confidence is True
    assert script_profile("hello").low_confidence is False


def test_characters_can_match_multiple_script_lists():
    assert scripts_for_char("[") == {"Common"}
    assert scripts_for_char("×") == {"Common", "Latin"}
    assert scripts_for_char("\u064e") == {"Arabic", "Inherited"}


def test_character_gate_accepts_any_selected_matching_script():
    assert not character_gate("\u064e", ScoreConfig(keep_scripts={"Arabic"}, languages=())).blocked
    assert not character_gate("\u064e", ScoreConfig(keep_scripts={"Inherited"}, languages=())).blocked


def test_unsupported_script_is_not_mislabeled_common():
    assert scripts_for_char("ა") == {"Unknown"}


def test_all_expands_within_each_category():
    assert normalize_scripts(["all"]) == set(script_lists())
    assert normalize_syntax_lists(["all"]) == set(syntax_lists())
    assert normalize_symbol_lists(["all"]) == set(symbol_lists())


def test_list_options_come_from_json_filenames():
    script_names = {
        item.name.removesuffix(".json")
        for item in resources.files("vocabpruner.resources.script_lists").iterdir()
        if item.name.endswith(".json")
    }
    syntax_names = {
        item.name.removesuffix(".json")
        for item in resources.files("vocabpruner.resources.syntax_lists").iterdir()
        if item.name.endswith(".json")
    }
    symbol_names = {
        item.name.removesuffix(".json")
        for item in resources.files("vocabpruner.resources.symbol_lists").iterdir()
        if item.name.endswith(".json")
    }

    assert set(script_lists()) == script_names
    assert set(syntax_lists()) == syntax_names
    assert set(symbol_lists()) == symbol_names


def test_syntax_and_symbol_detection():
    assert matching_syntax_lists("{", {"json"}) == ["json"]
    assert matching_symbol_lists("π", {"mathematics"}) == ["mathematics"]
    assert matching_symbol_lists("✅", {"emoji"}) == ["emoji"]


def test_json_words_match_as_complete_words():
    assert matching_syntax_word_lists(" keindahan", {"json"}) == []
    assert matching_syntax_word_lists("true", {"json"}) == ["json"]
    assert matching_syntax_word_lists(" false", {"json"}) == ["json"]
    assert matching_syntax_word_lists("null,", {"json"}) == ["json"]


def test_json_list_can_encode_whitespace(tmp_path):
    path = tmp_path / "whitespace.json"
    path.write_text(json.dumps({"characters": "\n\t", "words": [], "ranges": []}))

    definition = _load_list(path)

    assert definition.characters == frozenset({"\n", "\t"})


def test_lists_load_characters_words_and_ranges():
    assert "{" in syntax_lists()["coding"].characters
    assert "\\" in syntax_lists()["shell"].characters
    assert "#" in syntax_lists()["markdown"].characters
    assert {"true", "false", "null"} <= syntax_lists()["json"].words
    assert "+" in symbol_lists()["mathematics"].characters
    assert "=" in symbol_lists()["mathematics"].characters
    assert "π" in symbol_lists()["mathematics"].characters
    assert symbol_lists()["emoji"].ranges


def test_documented_list_sizes():
    def codepoints(definition):
        values = {ord(char) for char in definition.characters}
        for start, end in definition.ranges:
            values.update(range(start, end + 1))
        return values

    syntax = syntax_lists()
    symbols = symbol_lists()

    assert {
        name: len(item.characters) + sum(end - start + 1 for start, end in item.ranges)
        for name, item in script_lists().items()
        if item.ranges
    } == {
        "Arabic": 400,
        "Cyrillic": 304,
        "Devanagari": 128,
        "Greek": 400,
        "Han": 28096,
        "Hangul": 11536,
        "Hebrew": 112,
        "Hiragana": 96,
        "Katakana": 112,
        "Latin": 708,
        "Thai": 128,
    }

    assert {name: len(item.characters) + len(item.words) for name, item in syntax.items()} == {
        "coding": 32,
        "json": 23,
        "markdown": 16,
        "shell": 22,
        "xml": 15,
        "yaml": 20,
    }
    assert {name: len(codepoints(item)) for name, item in symbols.items()} == {
        "ascii-punctuation": 32,
        "emoji": 3523,
        "mathematics": 74,
    }
    assert len(set().union(*(item.characters for item in syntax.values()))) + len(
        set().union(*(item.words for item in syntax.values()))
    ) == 45
    assert len(set().union(*(codepoints(item) for item in symbols.values()))) == 3619
