import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from vocabpruner.gguf import GGUFTokenizer, config_dict, resolve_gguf, rewrite


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

    _, transformed = rewrite(source, output, [0, 2])
    reader = gguf.GGUFReader(output)

    assert transformed == ("token_embd.weight", "output.weight")
    assert reader.get_field("tokenizer.ggml.tokens").contents() == ["a", "ab"]
    assert reader.get_field("tokenizer.ggml.merges").contents() == []
    assert reader.get_field("tiny.vocab_size").contents() == 2
    assert reader.get_field("tokenizer.ggml.eos_token_id") is None
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert tensors["token_embd.weight"].data.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert tensors["blk.0.weight"].data.tolist() == [[1.0, 1.0], [1.0, 1.0]]
