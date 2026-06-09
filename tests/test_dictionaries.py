from tokenpruner.dictionaries import dictionary_matches, load_builtin_dictionaries


def test_dictionary_exact_and_subword_matches():
    dictionaries = {"english": {"hello"}, "programming": {"tokenizer"}}

    exact = dictionary_matches("hello", dictionaries)
    subword = dictionary_matches("he", dictionaries)
    suffix = dictionary_matches("izer", dictionaries)

    assert exact[0].mode == "exact"
    assert subword[0].term == "hello"
    assert suffix[0].term == "tokenizer"


def test_builtin_dictionaries_load():
    dictionaries = load_builtin_dictionaries()

    assert "english" in dictionaries
    assert "hello" in dictionaries["english"]
