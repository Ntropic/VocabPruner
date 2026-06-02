from tokenpruner.scripts import script_profile
from tokenpruner.symbols import SYMBOL_PRESETS, contains_emoji_or_pictograph, matching_symbol_presets, normalize_presets


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


def test_symbol_and_emoji_detection():
    assert "json" in matching_symbol_presets("{", {"json"})
    assert contains_emoji_or_pictograph("✅") is True


def test_symbol_preset_all_expands_to_every_builtin_preset():
    assert normalize_presets(["all"]) == set(SYMBOL_PRESETS)
    assert normalize_presets(["coding", "all"]) == set(SYMBOL_PRESETS)
