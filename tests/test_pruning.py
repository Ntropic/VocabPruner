import json

import pytest

from vocabpruner.architecture import inspect_model_architecture
from vocabpruner.pruning import (
    PruneRefusal,
    model_vocabulary_size,
    prune_model_and_tokenizer,
    prune_safetensors_checkpoint,
    remap_generation_config,
    remap_model_config,
    slice_model_vocab,
    validate_pruning_architecture,
    validate_pruning_runtime,
)
from vocabpruner.vocab import VocabRewriteError, rewrite_tokenizer_json


def test_validate_pruning_allows_shared_mtp_and_encoders():
    report = inspect_model_architecture(
        {
            "model_type": "fake",
            "text_config": {
                "vocab_size": 16,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
            },
            "vision_config": {"model_type": "fake_vision"},
        }
    )

    validate_pruning_architecture(report, 12)

    assert model_vocabulary_size(report, 12) == 16


def test_remap_generation_config_updates_token_ids(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text(
        json.dumps(
            {
                "bos_token_id": 0,
                "eos_token_id": [2, 0],
                "pad_token_id": 2,
                "max_length": 32,
            }
        ),
        encoding="utf-8",
    )

    remap_generation_config(path, {0: 0, 2: 1}, 4)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "bos_token_id": 0,
        "eos_token_id": [1, 0],
        "max_length": 32,
        "pad_token_id": 1,
    }


def test_validate_pruning_refuses_differently_sized_vocab_surfaces():
    report = inspect_model_architecture(
        {
            "model_type": "fake",
            "text_config": {"vocab_size": 8},
            "talker_config": {"vocab_size": 4},
        }
    )

    with pytest.raises(PruneRefusal, match="smaller than the tokenizer"):
        validate_pruning_architecture(report, 8)


def test_validate_pruning_runtime_names_missing_quantization_dependency(monkeypatch):
    report = inspect_model_architecture(
        {
            "model_type": "fake",
            "vocab_size": 8,
            "quantization_config": {"quant_method": "modelopt"},
        }
    )
    available = {"torch": True, "modelopt": False}
    monkeypatch.setattr("vocabpruner.pruning._module_available", lambda name: available.get(name, False))

    with pytest.raises(PruneRefusal, match="nvidia-modelopt"):
        validate_pruning_runtime(report)


def test_rewrite_tokenizer_json_remaps_dict_vocab_and_added_tokens(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"<s>": 0, "h": 1, "i": 2, "hi": 3, "bye": 4},
                    "merges": ["h i", "b ye"],
                },
                "added_tokens": [{"id": 0, "content": "<s>"}, {"id": 4, "content": "bye"}],
                "post_processor": {"special_tokens": {"<s>": {"ids": ["<s>", 0], "tokens": ["<s>"]}}},
            }
        ),
        encoding="utf-8",
    )

    rewrite_tokenizer_json(tokenizer_json, {0: 0, 1: 1, 2: 2, 3: 3})

    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    assert data["model"]["vocab"] == {"<s>": 0, "h": 1, "i": 2, "hi": 3}
    assert data["model"]["merges"] == ["h i"]
    assert data["added_tokens"] == [{"id": 0, "content": "<s>"}]
    assert data["post_processor"]["special_tokens"]["<s>"]["ids"] == ["<s>", 0]


def test_rewrite_tokenizer_json_remaps_list_vocab(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps({"model": {"type": "Unigram", "vocab": [["<s>", 0.0], ["hello", -1.0], ["bye", -2.0]]}}),
        encoding="utf-8",
    )

    rewrite_tokenizer_json(tokenizer_json, {0: 0, 2: 1})

    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    assert data["model"]["vocab"] == [["<s>", 0.0], ["bye", -2.0]]


def test_rewrite_tokenizer_json_refuses_unknown_vocab_shape(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(json.dumps({"model": {"type": "Mystery"}}), encoding="utf-8")

    with pytest.raises(VocabRewriteError, match="model.vocab"):
        rewrite_tokenizer_json(tokenizer_json, {0: 0})


def test_slice_model_vocab_slices_input_and_untied_output_head():
    torch = pytest.importorskip("torch")

    class Config:
        vocab_size = 4

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.input_embeddings = torch.nn.Embedding(4, 2)
            self.output_embeddings = torch.nn.Linear(2, 4, bias=False)
            self.input_embeddings.weight.data.copy_(torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
            self.output_embeddings.weight.data.copy_(
                torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]])
            )

        def get_input_embeddings(self):
            return self.input_embeddings

        def set_input_embeddings(self, embeddings):
            self.input_embeddings = embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

        def set_output_embeddings(self, embeddings):
            self.output_embeddings = embeddings

    model = FakeModel()

    slice_model_vocab(model, [0, 2, 3], {0: 0, 2: 1, 3: 2})

    assert model.config.vocab_size == 3
    assert model.get_input_embeddings().weight.tolist() == [[0.0, 1.0], [4.0, 5.0], [6.0, 7.0]]
    assert model.get_output_embeddings().weight.tolist() == [[10.0, 11.0], [14.0, 15.0], [16.0, 17.0]]


def test_slice_model_vocab_handles_shared_and_separate_mtp_surfaces():
    torch = pytest.importorskip("torch")

    class Config:
        vocab_size = 4

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
            self.mtp = torch.nn.Module()
            self.mtp.embed_tokens = torch.nn.Embedding(4, 2)
            self.mtp.embed_tokens.weight = self.embed_tokens.weight
            self.mtp.lm_head = torch.nn.Linear(2, 4, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    model = FakeModel()

    surfaces = slice_model_vocab(model, [0, 2], {0: 0, 2: 1}, original_vocab_size=4)

    assert model.embed_tokens.weight.shape == (2, 2)
    assert model.mtp.embed_tokens.weight is model.embed_tokens.weight
    assert model.lm_head.weight.shape == (2, 2)
    assert model.mtp.lm_head.weight.shape == (2, 2)
    assert {surface.kind for surface in surfaces} == {"embedding", "output_head", "mtp_embedding", "mtp_output_head"}
    assert next(surface for surface in surfaces if surface.kind == "mtp_embedding").shared_with == "input_embeddings"


def test_slice_model_vocab_preserves_quantized_module_and_row_metadata():
    torch = pytest.importorskip("torch")

    class QuantizedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.uint8).reshape(4, 4), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2), requires_grad=False)
            self.out_features = 4

    class Config:
        vocab_size = 4

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = QuantizedHead()

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    model = FakeModel()
    original_type = type(model.lm_head)

    surfaces = slice_model_vocab(model, [1, 3], {1: 0, 3: 1}, original_vocab_size=4)

    assert type(model.lm_head) is original_type
    assert model.lm_head.weight.tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]
    assert model.lm_head.weight_scale.tolist() == [[2.0, 3.0], [6.0, 7.0]]
    assert model.lm_head.out_features == 2
    assert next(surface for surface in surfaces if surface.kind == "output_head").quantized is True


def test_prune_model_and_tokenizer_copies_source_sidecars(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    class Config:
        vocab_size = 4

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.embed_tokens = torch.nn.Embedding(4, 2)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return None

        def save_pretrained(self, output_dir):
            (output_dir / "pytorch_model.bin").write_bytes(b"pruned")

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            (output_dir / "tokenizer.json").write_text(
                json.dumps({"model": {"type": "BPE", "vocab": {"<s>": 0, "hello": 1, "bye": 2, "!": 3}}}),
                encoding="utf-8",
            )

    source = tmp_path / "source"
    source.mkdir()
    (source / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<s>"}), encoding="utf-8")
    (source / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (source / "processor").mkdir()
    (source / "processor" / "config.json").write_text("{}", encoding="utf-8")
    (source / "processor" / "pytorch_model.bin").write_bytes(b"nested weights")
    (source / "README.md").write_text("model card", encoding="utf-8")
    (source / "modelopt_state.pth").write_bytes(b"modelopt state")
    (source / "model.safetensors").write_bytes(b"source weights")
    (source / "raw_state.pth").write_bytes(b"raw weights")
    monkeypatch.setattr("vocabpruner.vocab.validate_saved_tokenizer", lambda *args, **kwargs: None)

    prune_model_and_tokenizer(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        output_dir=tmp_path / "out",
        retained_token_ids=[0, 2],
        original_vocab_size=4,
        model_vocab_size=4,
        source_dir=source,
    )

    assert json.loads((tmp_path / "out" / "special_tokens_map.json").read_text(encoding="utf-8")) == {
        "bos_token": "<s>"
    }
    assert (tmp_path / "out" / "chat_template.jinja").read_text(encoding="utf-8") == "{{ messages }}"
    assert (tmp_path / "out" / "processor" / "config.json").exists()
    assert (tmp_path / "out" / "README.md").read_text(encoding="utf-8") == "model card"
    assert (tmp_path / "out" / "modelopt_state.pth").read_bytes() == b"modelopt state"
    assert not (tmp_path / "out" / "processor" / "pytorch_model.bin").exists()
    assert not (tmp_path / "out" / "model.safetensors").exists()
    assert not (tmp_path / "out" / "raw_state.pth").exists()


def test_remap_model_config_updates_text_vocab_and_modality_ids():
    config = {
        "text_config": {
            "vocab_size": 8,
            "bos_token_id": 1,
            "eos_token_id": 7,
            "mtp_use_dedicated_embeddings": False,
        },
        "vision_config": {"hidden_size": 32},
        "image_token_id": 6,
        "video_token_id": 7,
    }

    remap_model_config(config, {1: 0, 6: 1, 7: 2}, original_vocab_size=8)

    assert config["text_config"]["vocab_size"] == 3
    assert config["text_config"]["bos_token_id"] == 0
    assert config["text_config"]["eos_token_id"] == 2
    assert config["image_token_id"] == 1
    assert config["video_token_id"] == 2
    assert config["vision_config"]["hidden_size"] == 32


def test_prune_safetensors_checkpoint_streams_only_vocab_tensors(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            path = output_dir / "tokenizer.json"
            path.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {"<s>": 0, "hello": 1, "bye": 2, "!": 3},
                            "merges": [],
                        },
                        "added_tokens": [{"id": 0, "content": "<s>"}],
                    }
                ),
                encoding="utf-8",
            )

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"vocab_size": 4}), encoding="utf-8")
    (source / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 0,
                "eos_token_id": [2, 0],
                "pad_token_id": 2,
                "max_length": 32,
            }
        ),
        encoding="utf-8",
    )
    (source / "README.md").write_text("model card", encoding="utf-8")
    (source / "modelopt_state.pth").write_bytes(b"modelopt state")
    (source / "raw_state.pth").write_bytes(b"raw weights")
    safetensors.save_file(
        {
            "model.embed_tokens.weight": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.float32),
            "lm_head.weight": torch.tensor([[10, 11], [12, 13], [14, 15], [16, 17]], dtype=torch.float32),
            "lm_head.weight_scale": torch.tensor([[1], [2], [3], [4]], dtype=torch.float32),
            "model.layers.0.mlp.down_proj.weight": torch.tensor([[100, 101], [102, 103]], dtype=torch.float32),
        },
        source / "model.safetensors",
    )
    safetensors.save_file(
        {
            "mtp.fc.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        },
        source / "model_mtp.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                    "lm_head.weight_scale": "model.safetensors",
                    "model.layers.0.mlp.down_proj.weight": "model.safetensors",
                    "mtp.fc.weight": "model_mtp.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("vocabpruner.vocab.validate_saved_tokenizer", lambda *args, **kwargs: None)

    result = prune_safetensors_checkpoint(
        model_source=str(source),
        tokenizer=FakeTokenizer(),
        config={"vocab_size": 4},
        output_dir=tmp_path / "out",
        retained_token_ids=[0, 2],
        original_vocab_size=4,
        model_vocab_size=4,
    )

    assert result is not None
    tensors = safetensors.load_file(tmp_path / "out" / "model.safetensors")
    mtp_tensors = safetensors.load_file(tmp_path / "out" / "model_mtp.safetensors")
    assert tensors["model.embed_tokens.weight"].tolist() == [[0, 1], [4, 5]]
    assert tensors["lm_head.weight"].tolist() == [[10, 11], [14, 15]]
    assert tensors["lm_head.weight_scale"].tolist() == [[1], [3]]
    assert tensors["model.layers.0.mlp.down_proj.weight"].tolist() == [[100, 101], [102, 103]]
    assert mtp_tensors["mtp.fc.weight"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    config = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    generation_config = json.loads(
        (tmp_path / "out" / "generation_config.json").read_text(encoding="utf-8")
    )
    report = json.loads((tmp_path / "out" / "pruning_report.json").read_text(encoding="utf-8"))
    assert config["vocab_size"] == 2
    assert generation_config == {
        "bos_token_id": 0,
        "eos_token_id": [1, 0],
        "max_length": 32,
        "pad_token_id": 1,
    }
    assert (tmp_path / "out" / "README.md").read_text(encoding="utf-8") == "model card"
    assert (tmp_path / "out" / "modelopt_state.pth").read_bytes() == b"modelopt state"
    assert not (tmp_path / "out" / "raw_state.pth").exists()
    surface_names = {surface.name for surface in result.transformed_surfaces}
    assert surface_names == {"model.embed_tokens", "lm_head"}
    assert result.original_checkpoint_bytes == (source / "model.safetensors").stat().st_size + (
        source / "model_mtp.safetensors"
    ).stat().st_size
    assert result.pruned_checkpoint_bytes == (
        tmp_path / "out" / "model.safetensors"
    ).stat().st_size + (tmp_path / "out" / "model_mtp.safetensors").stat().st_size
    assert report["original_checkpoint_bytes"] == result.original_checkpoint_bytes
    assert report["pruned_checkpoint_bytes"] == result.pruned_checkpoint_bytes
    assert report["checkpoint_size_reduction_bytes"] == result.original_checkpoint_bytes - result.pruned_checkpoint_bytes


def test_prune_safetensors_checkpoint_refuses_missing_indexed_file(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {"model.embed_tokens.weight": torch.zeros((4, 2), dtype=torch.float32)},
        source / "model.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "mtp.fc.weight": "model_mtp.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PruneRefusal, match="snapshot is incomplete"):
        prune_safetensors_checkpoint(
            model_source=str(source),
            tokenizer=object(),
            config={"vocab_size": 4},
            output_dir=tmp_path / "out",
            retained_token_ids=[0, 1],
            original_vocab_size=4,
            model_vocab_size=4,
        )


def test_prune_safetensors_checkpoint_refuses_empty_indexed_file(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {"model.embed_tokens.weight": torch.zeros((4, 2), dtype=torch.float32)},
        source / "model.safetensors",
    )
    (source / "model_mtp.safetensors").write_bytes(b"")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "mtp.fc.weight": "model_mtp.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PruneRefusal, match="file\\(s\\) are empty"):
        prune_safetensors_checkpoint(
            model_source=str(source),
            tokenizer=object(),
            config={"vocab_size": 4},
            output_dir=tmp_path / "out",
            retained_token_ids=[0, 1],
            original_vocab_size=4,
            model_vocab_size=4,
        )


def test_prune_safetensors_checkpoint_refuses_corrupt_indexed_file(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {"model.embed_tokens.weight": torch.zeros((4, 2), dtype=torch.float32)},
        source / "model.safetensors",
    )
    (source / "model_mtp.safetensors").write_bytes(b"not a safetensors file")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "mtp.fc.weight": "model_mtp.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PruneRefusal, match="incomplete or corrupt"):
        prune_safetensors_checkpoint(
            model_source=str(source),
            tokenizer=object(),
            config={"vocab_size": 4},
            output_dir=tmp_path / "out",
            retained_token_ids=[0, 1],
            original_vocab_size=4,
            model_vocab_size=4,
        )
