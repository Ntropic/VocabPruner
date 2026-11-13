import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from vocabpruner.tokenization.gguf import GGUFTokenizer
from vocabpruner.checkpoints.architecture import config_dict
from vocabpruner.storage.gguf import resolve_gguf
from vocabpruner.checkpoints.gguf import prune_gguf_checkpoint, rewrite


def _tiny_gguf(path):
    writer = gguf.GGUFWriter(path, "tiny")
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "ab"])
    writer.add_array("tokenizer.ggml.token_type", [1, 3, 1])
    writer.add_array("tokenizer.ggml.scores", [1.0, 2.0, 3.0])
    writer.add_array("tokenizer.ggml.merges", ["a b"])
    writer.add_uint32("tokenizer.ggml.eos_token_id", 1)
    writer.add_uint32("tiny.vocab_size", 3)
    writer.add_tensor("token_embd.weight", np.arange(6, dtype=np.float32).reshape(3, 2))
    writer.add_tensor("output.weight", np.arange(6, dtype=np.float32).reshape(3, 2))
    writer.add_tensor("blk.0.weight", np.ones((2, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_tokenizer_and_config(tmp_path):
    source = tmp_path / "tiny.gguf"
    _tiny_gguf(source)

    tokenizer = GGUFTokenizer(source)

    assert tokenizer.get_vocab() == {"a": 0, "b": 1, "ab": 2}
    assert tokenizer.all_special_ids == [1]
    assert config_dict(source)["vocab_size"] == 3


def test_gguf_pruning_backend_writes_maps_and_validated_checkpoint(tmp_path):
    source = tmp_path / "tiny.gguf"
    _tiny_gguf(source)
    output = tmp_path / "output"
    result = prune_gguf_checkpoint(
        model_source=str(source), output_dir=output,
        retained_token_ids=[0], original_vocab_size=3,
    )
    assert result is not None
    assert result.old_to_new == {0: 0, 1: 1}
    assert result.retained_count == 2
    assert (output / "pruning_report.json").is_file()
    assert (output / "old_to_new_token_ids.json").is_file()
    assert (output / "tiny-pruned.tokenizer_validation.json").is_file()
    assert GGUFTokenizer(output / "tiny-pruned.gguf").tokens == ["a", "b"]


def test_hf_id_resolves_unique_local_gguf(monkeypatch, tmp_path):
    directory = tmp_path / "models--unsloth--Qwen3.8-27B-GGUF" / "snapshots" / "abc123"
    directory.mkdir(parents=True)
    source = directory / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    source.touch()
    (directory.parents[1] / "refs").mkdir()
    (directory.parents[1] / "refs" / "main").write_text("abc123")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert resolve_gguf("unsloth/Qwen3.8-27B-GGUF") == source


def test_hf_id_refuses_ambiguous_local_ggufs(monkeypatch, tmp_path):
    directory = tmp_path / "models--org--model" / "snapshots" / "abc123"
    directory.mkdir(parents=True)
    (directory / "q4.gguf").touch()
    (directory / "q8.gguf").touch()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    with pytest.raises(ValueError, match="Multiple GGUF files"):
        resolve_gguf("org/model")


def test_gguf_rewrite_slices_vocab_metadata_and_tensors(tmp_path):
    source = tmp_path / "tiny.gguf"
    output = tmp_path / "pruned.gguf"
    _tiny_gguf(source)

    _, transformed = rewrite(source, output, [0, 1])
    reader = gguf.GGUFReader(output)

    assert transformed == ("token_embd.weight", "output.weight")
    assert reader.get_field("tokenizer.ggml.tokens").contents() == ["a", "b"]
    assert reader.get_field("tokenizer.ggml.merges").contents() == []
    assert reader.get_field("tiny.vocab_size").contents() == 2
    assert reader.get_field("tokenizer.ggml.eos_token_id").contents() == 1
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert tensors["token_embd.weight"].data.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert tensors["blk.0.weight"].data.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_gguf_preserves_byte_alphabet_and_validates_every_retained_token(tmp_path):
    import json
    from tokenizers.pre_tokenizers import ByteLevel
    from vocabpruner.errors import VocabRewriteError
    from vocabpruner.tokenization.preservation import complete_retained_token_ids

    source = tmp_path / "bytes.gguf"
    output = tmp_path / "bytes-pruned.gguf"
    tokens = ["discard", *sorted(ByteLevel.alphabet()), "Ã¤"]
    writer = gguf.GGUFWriter(source, "tiny")
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", ["Ã ¤"])
    writer.add_array("tokenizer.ggml.scores", [0.0] * len(tokens))
    writer.add_tensor("token_embd.weight", np.arange(len(tokens) * 2, dtype=np.float32).reshape(len(tokens), 2))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises((ValueError, VocabRewriteError), match="omits required"):
        rewrite(source, output, [257])
    retained = complete_retained_token_ids(GGUFTokenizer(source), [257])
    rewrite(source, output, retained)
    saved = GGUFTokenizer(output)
    assert saved.decode([saved.get_vocab()["Ã¤"]]) == "ä"
    assert set(ByteLevel.alphabet()) <= saved.get_vocab().keys()
    report = json.loads(output.with_suffix(".tokenizer_validation.json").read_text())
    assert report["retained_tokens_checked"] == len(retained)
    assert report["byte_tokens_preserved"] == 256
