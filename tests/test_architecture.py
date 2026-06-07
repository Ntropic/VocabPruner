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
    assert "resize every affected embedding and output head" in output


def test_mtp_vocabulary_does_not_imply_a_separate_lm_head():
    config = {"mtp_config": {"vocab_size": 4096, "tie_word_embeddings": False}}

    report = inspect_model_architecture(config)

    assert report.mtp_head_status == "separate vocabulary declared; head sharing unknown"
