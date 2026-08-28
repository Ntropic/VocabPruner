from vocabpruner.vocab import is_byte_fallback_token, tokenizer_records, vocabulary_stats


class TinyTokenizer:
    all_special_ids = [0]
    all_special_tokens = ["<s>"]
    special_tokens_map = {"bos_token": "<s>"}

    def get_vocab(self):
        return {"<s>": 0, "hello": 1, "<0x20>": 2}

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        values = {0: "<s>", 1: "hello", 2: " "}
        return values[token_ids[0]]


def test_tokenizer_records_extract_metadata():
    records = tokenizer_records(TinyTokenizer())

    assert records[0].is_special is True
    assert records[1].decoded_text == "hello"
    assert records[2].is_byte_fallback is True


def test_vocabulary_stats():
    stats = vocabulary_stats(tokenizer_records(TinyTokenizer()))

    assert stats["vocab_size"] == 3
    assert stats["special_tokens"] == 1
    assert stats["byte_fallback_tokens"] == 1


def test_byte_fallback_detection():
    assert is_byte_fallback_token("<0xFF>") is True
