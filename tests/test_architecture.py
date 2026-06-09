from tokenpruner.architecture import inspect_model_architecture, render_model_architecture


class FakeOmniConfig:
    model_type = "fake_omni"

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "architectures": ["FakeOmniForConditionalGeneration"],
            "text_config": {
                "vocab_size": 128000,
                "tie_word_embeddings": False,
                "num_nextn_predict_layers": 2,
            },
            "vision_config": {"model_type": "fake_vision"},
            "audio_encoder_config": {"model_type": "fake_audio"},
            "talker_config": {
                "vocab_size": 4096,
                "codebook_size": 2048,
                "tie_word_embeddings": True,
            },
        }


def test_architecture_inspection_finds_mtp_vocabularies_and_encoders():
    report = inspect_model_architecture(FakeOmniConfig())

    assert report.has_mtp is True
    assert report.mtp_head_status == "not declared; inspect model modules or checkpoint tensors"
    assert ("text_config.vocab_size", 128000) in report.vocabulary_surfaces
    assert ("talker_config.vocab_size", 4096) in report.vocabulary_surfaces
    assert "vision_config" in report.encoders
    assert "audio_encoder_config" in report.encoders
    assert ("text_config", False) in report.embedding_ties


def test_architecture_render_warns_about_multiple_pruning_surfaces():
    output = render_model_architecture(inspect_model_architecture(FakeOmniConfig()))

    assert "MTP: yes" in output
    assert "Separate MTP LM Head: not declared" in output
    assert "MTP Embeddings: not declared" in output
    assert "resize every affected embedding and output head" in output


def test_mtp_vocabulary_does_not_imply_a_separate_lm_head():
    config = {"mtp_config": {"vocab_size": 4096, "tie_word_embeddings": False}}

    report = inspect_model_architecture(config)

    assert report.mtp_head_status == "separate vocabulary declared; head sharing unknown"


def test_architecture_inspection_accepts_integer_config_keys():
    config = {
        "model_type": "fake",
        "vocab_size": 128,
        "quantization_config": {
            "modules": {
                0: {"bits": 4},
                1: {"bits": 4},
            }
        },
    }

    report = inspect_model_architecture(config)

    assert report.model_type == "fake"
    assert report.vocabulary_surfaces == (("vocab_size", 128),)
    assert report.encoders == ()


def test_qwen35_nvfp4_reports_shared_mtp_embeddings_and_quantization():
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False,
        },
        "vision_config": {"model_type": "qwen3_5_moe_vision"},
        "quantization_config": {"quant_method": "modelopt"},
    }

    report = inspect_model_architecture(config)

    assert report.has_mtp is True
    assert report.mtp_dedicated_embeddings is False
    assert report.quantization_method == "modelopt"
    assert report.vocabulary_surfaces == (("text_config.vocab_size", 248320),)
    assert report.encoders == ("vision_config",)


def test_unsloth_qwen35_nvfp4_reports_compressed_tensors():
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False,
        },
        "vision_config": {"model_type": "qwen3_5"},
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "ignore": ["lm_head", "re:^mtp\\..*"],
        },
    }

    report = inspect_model_architecture(config)

    assert report.has_mtp is True
    assert report.mtp_dedicated_embeddings is False
    assert report.quantization_method == "compressed-tensors"
