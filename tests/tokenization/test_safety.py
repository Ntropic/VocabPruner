import json

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, Tokenizer
from transformers import PreTrainedTokenizerFast
import pytest

from vocabpruner.errors import VocabRewriteError
from vocabpruner.mapping import compact_token_maps
from vocabpruner.tokenization.loading import tokenizer_records
from vocabpruner.tokenization.preservation import (
    complete_retained_token_ids,
    required_byte_token_ids,
    tokenizer_data,
)
from vocabpruner.tokenization.rewriting import save_remapped_tokenizer
from vocabpruner.tokenization.validation import (
    validate_saved_tokenizer,
    validate_tokenizer_vocabulary,
)


def byte_tokenizer():
    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    merges = [("Ã", "¤"), ("Ã", "¶"), ("Ã", "¼"), ("Ã", "Ł"), ("Ã", "©"),
              ("â", "Ģ"), ("âĢ", "ľ"), ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    vocab["<s>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.normalizer = normalizers.NFC()
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>")


def test_all_bytes_and_transitive_merge_dependencies_survive(tmp_path):
    original = byte_tokenizer()
    vocab = original.get_vocab()
    records = tokenizer_records(original)
    assert sum(record.is_byte_fallback for record in records) == 256
    retained = complete_retained_token_ids(original, [vocab["hello"], vocab["âĢľ"]])
    assert {vocab[token] for token in ["he", "hel", "hell", "âĢ", "<s>"]} <= set(retained)
    assert vocab["Ã¤"] not in retained
    mapping, _ = compact_token_maps(retained)
    save_remapped_tokenizer(original, tmp_path, mapping)
    saved = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    for text in ["ä ö ü ß é", "Straße für schön", "“Hello”—10€…", "中文 العربية русский", "🙂", "a\x00b"]:
        assert saved.decode(saved.encode(text, add_special_tokens=False)) == text
    assert saved.encode("hello", add_special_tokens=False) == [mapping[vocab["hello"]]]
    report = json.loads((tmp_path / "tokenizer_validation.json").read_text())
    assert report["status"] == "passed"
    assert report["retained_tokens_checked"] == len(mapping)
    assert report["byte_tokens_preserved"] == 256


def test_export_rejects_incomplete_byte_map(tmp_path):
    tokenizer = byte_tokenizer()
    mapping, _ = compact_token_maps([tokenizer.get_vocab()["Ã¤"]])
    with pytest.raises(VocabRewriteError, match="omits required"):
        save_remapped_tokenizer(tokenizer, tmp_path, mapping)
    assert not (tmp_path / "tokenizer.json").exists()


def test_post_validation_detects_character_loss(tmp_path):
    original = byte_tokenizer()
    mapping, _ = compact_token_maps(complete_retained_token_ids(original, [0]))
    save_remapped_tokenizer(original, tmp_path, mapping)
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text())
    data["normalizer"] = {"type": "Replace", "pattern": {"String": "ä"}, "content": ""}
    path.write_text(json.dumps(data))
    with pytest.raises(VocabRewriteError, match="normalizer"):
        validate_saved_tokenizer(tmp_path, len(mapping), source_tokenizer=original, old_to_new=mapping)
    assert not (tmp_path / "tokenizer_validation.json").exists()


@pytest.mark.parametrize("model_type", ["Custom", "Unknown"])
def test_unsupported_tokenizers_are_refused(model_type):
    with pytest.raises(VocabRewriteError, match="Unsupported tokenizer format"):
        required_byte_token_ids({"model": {"type": model_type, "vocab": {}}})


def test_explicit_byte_fallback_is_complete_and_protected(tmp_path):
    vocab = {f"<0x{value:02X}>": value for value in range(256)}
    vocab.update({"<unk>": 256, "a": 257, "b": 258, "ab": 259})
    backend = Tokenizer(models.BPE(vocab, [("a", "b")], unk_token="<unk>", byte_fallback=True))
    backend.decoder = decoders.ByteFallback()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    retained = complete_retained_token_ids(tokenizer, [257])
    assert set(range(257)) <= set(retained)
    mapping, _ = compact_token_maps(retained)
    save_remapped_tokenizer(tokenizer, tmp_path, mapping)
    data = tokenizer_data(tokenizer)
    del data["model"]["vocab"]["<0xFF>"]
    with pytest.raises(VocabRewriteError, match="all 256"):
        required_byte_token_ids(data)


@pytest.mark.parametrize("kind", ["BPE", "Unigram", "WordPiece", "WordLevel", "Character"])
def test_supported_tokenizer_vocabularies_are_validated_exhaustively(tmp_path, kind):
    vocab = {"discard": 0, "<unk>": 1, "a": 2, "b": 3, "ab": 4, "##b": 5}
    if kind == "Unigram":
        model = models.Unigram([(token, -float(index)) for token, index in vocab.items()], unk_id=1)
    elif kind == "BPE":
        model = models.BPE(vocab, [("a", "b")], unk_token="<unk>")
    elif kind == "WordPiece":
        model = models.WordPiece(vocab, unk_token="<unk>")
    else:
        model = models.WordLevel(vocab, unk_token="<unk>")
    backend = Tokenizer(model)
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated") if kind == "Character" else pre_tokenizers.Whitespace()
    if kind == "WordPiece":
        backend.decoder = decoders.WordPiece(prefix="##")
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    retained = complete_retained_token_ids(original, [4])
    assert 1 in retained
    mapping, _ = compact_token_maps(retained)
    save_remapped_tokenizer(original, tmp_path, mapping)
    saved = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    assert "discard" not in saved.get_vocab()
    report = validate_tokenizer_vocabulary(original, saved, mapping)
    assert report["retained_tokens_checked"] == len(mapping)
    if kind == "Unigram":
        assert tokenizer_data(saved)["model"]["unk_id"] == mapping[1]


def test_unigram_byte_fallback_and_scores_survive(tmp_path):
    entries = [("discard", -5.0), ("<unk>", 0.0)]
    entries += [(f"<0x{value:02X}>", -10.0) for value in range(256)]
    entries += [("ä", -1.5)]
    backend = Tokenizer(models.Unigram(entries, unk_id=1, byte_fallback=True))
    backend.decoder = decoders.ByteFallback()
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    mapping, _ = compact_token_maps(complete_retained_token_ids(original, [258]))
    save_remapped_tokenizer(original, tmp_path, mapping)
    saved = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    assert tokenizer_data(saved)["model"]["vocab"][-1] == ["ä", -1.5]
    assert len(required_byte_token_ids(tokenizer_data(saved))) == 256


def test_retained_token_decode_corruption_is_detected(tmp_path):
    original = byte_tokenizer()
    mapping, _ = compact_token_maps(complete_retained_token_ids(original, [0]))
    save_remapped_tokenizer(original, tmp_path, mapping)
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text())
    data["decoder"] = {"type": "Replace", "pattern": {"String": "a"}, "content": "x"}
    path.write_text(json.dumps(data))
    with pytest.raises(VocabRewriteError, match="Decoded token changed"):
        validate_saved_tokenizer(tmp_path, len(mapping), original, mapping)


def test_bpe_continuation_prefix_dependencies(tmp_path):
    vocab = {"<unk>": 0, "a": 1, "##b": 2, "ab": 3, "discard": 4}
    backend = Tokenizer(models.BPE(vocab, [("a", "##b")], unk_token="<unk>", continuing_subword_prefix="##"))
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    retained = complete_retained_token_ids(original, [3])
    assert retained == [0, 1, 2, 3]
    mapping, _ = compact_token_maps(retained)
    save_remapped_tokenizer(original, tmp_path, mapping)


def test_post_processor_special_ids_are_remapped(tmp_path):
    original = byte_tokenizer()
    special_id = original.get_vocab()["<s>"]
    original.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", special_id)],
    )
    mapping, _ = compact_token_maps(complete_retained_token_ids(original, [0]))
    save_remapped_tokenizer(original, tmp_path, mapping)
    saved = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    assert saved.encode("a")[0] == mapping[special_id]


def test_streaming_pruning_keeps_weights_aligned_with_completed_vocabulary(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    from vocabpruner.checkpoints.safetensors import prune_safetensors_checkpoint

    tokenizer = byte_tokenizer()
    count = len(tokenizer)
    source = tmp_path / "source"
    source.mkdir()
    tokenizer.save_pretrained(source)
    weights = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
    save_file({"model.embed_tokens.weight": weights, "lm_head.weight": weights.clone()}, source / "model.safetensors")
    result = prune_safetensors_checkpoint(
        model_source=str(source), tokenizer=tokenizer, config={"vocab_size": count},
        output_dir=tmp_path / "pruned", retained_token_ids=[tokenizer.get_vocab()["hello"]],
        original_vocab_size=count,
    )
    saved = load_file(tmp_path / "pruned/model.safetensors")
    expected = weights[sorted(result.old_to_new)]
    assert torch.equal(saved["model.embed_tokens.weight"], expected)
    assert torch.equal(saved["lm_head.weight"], expected)
    assert (tmp_path / "pruned/tokenizer_validation.json").is_file()


def test_unigram_score_corruption_is_detected(tmp_path):
    backend = Tokenizer(models.Unigram([("<unk>", 0.0), ("a", -1.0), ("b", -2.0)], unk_id=0))
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    mapping = {0: 0, 1: 1}
    save_remapped_tokenizer(original, tmp_path, mapping)
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text())
    data["model"]["vocab"][1][1] = -9.0
    path.write_text(json.dumps(data))
    with pytest.raises(VocabRewriteError, match="scores changed"):
        validate_saved_tokenizer(tmp_path, 2, original, mapping)


def test_wordpiece_sidecar_and_stale_sentencepiece_file(tmp_path):
    backend = Tokenizer(models.WordPiece({"discard": 0, "[UNK]": 1, "word": 2}, unk_token="[UNK]"))
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    (tmp_path / "vocab.txt").write_text("discard\n[UNK]\nword\n")
    (tmp_path / "tokenizer.model").write_bytes(b"old tokenizer")
    save_remapped_tokenizer(original, tmp_path, {1: 0, 2: 1})
    assert (tmp_path / "vocab.txt").read_text() == "[UNK]\nword\n"
    assert not (tmp_path / "tokenizer.model").exists()


def test_wrong_token_map_and_special_attributes_are_rejected(tmp_path):
    original = byte_tokenizer()
    mapping, _ = compact_token_maps(complete_retained_token_ids(original, [0]))
    save_remapped_tokenizer(original, tmp_path, mapping)
    saved = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    wrong = dict(mapping)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(VocabRewriteError, match="token map"):
        validate_tokenizer_vocabulary(original, saved, wrong)
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text())
    data["added_tokens"][0]["lstrip"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(VocabRewriteError, match="attributes changed"):
        validate_saved_tokenizer(tmp_path, len(mapping), original, mapping)
