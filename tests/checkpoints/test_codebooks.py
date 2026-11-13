import json

import pytest

from vocabpruner.analysis.scoring import ScoreConfig
from vocabpruner.checkpoints.codebooks import codebook_config_ids, codebook_tensor_axis
from vocabpruner.cli import build_parser
from vocabpruner.errors import PruneRefusal
from vocabpruner.models import PruneResult
from vocabpruner.presentation.artifacts import resolve_artifact_dir
from vocabpruner.storage.metadata import load_related_token_ids, resolve_related_model_dir
import vocabpruner.checkpoints.codebooks as checkpoints_codebooks
import vocabpruner.cli as cli
import vocabpruner.workflows as workflows


def test_codebook_rules_and_related_map(monkeypatch, tmp_path):
    assert codebook_tensor_axis("model.mtp.0.candidate_selector.predecessor_codebook", (4, 2), 4) == 0
    assert codebook_tensor_axis("mtp.0.markov_head.markov_w2.weight", (4, 2), 4) == 0
    assert codebook_tensor_axis("model.mtp.0.head.weight", (4, 2), 4) == 0
    assert codebook_config_ids(
        {"dflash_config": {"mask_token_id": 3}, "dspark_noise_token_id": 2}
    ) == {2, 3}
    monkeypatch.setattr(checkpoints_codebooks, "codebook_rules", lambda: ((("column_table", 1),), ()))
    assert codebook_tensor_axis("model.column_table", (2, 4), 4) == 1

    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "old_to_new_token_ids.json").write_text(
        json.dumps({"0": 0, "2": 1, "3": 2}), encoding="utf-8"
    )
    (reference / "pruning_report.json").write_text(
        json.dumps({"original_vocab_size": 4, "source_model_vocab_size": 4}), encoding="utf-8"
    )
    assert load_related_token_ids(reference, 4) == [0, 2, 3]
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"text_config": {"vocab_size": 4}}), encoding="utf-8"
    )
    (reference / "vocabpruner_model.json").write_text(
        json.dumps({"source_model": str(source)}), encoding="utf-8"
    )
    (reference / "pruning_report.json").write_text(
        json.dumps({"original_vocab_size": 3}), encoding="utf-8"
    )
    with pytest.raises(PruneRefusal, match="regenerate"):
        load_related_token_ids(reference, 4)
    (reference / "pruning_report.json").write_text(
        json.dumps({"original_vocab_size": 3, "source_model_vocab_size": 4}), encoding="utf-8"
    )
    assert load_related_token_ids(reference, 4) == [0, 2, 3]
    with pytest.raises(PruneRefusal, match="model has 5"):
        load_related_token_ids(reference, 5)


def test_related_options_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["prune", "draft", "--related-path", "reference", "--related-model", "target"]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["prune", "draft", "--save", "output", "--save-suffix", "latin"]
        )


def test_generated_suffix_and_related_model_name_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("VOCABPRUNER_SAVE_DIR", str(tmp_path))
    config = ScoreConfig()
    named = tmp_path / "target-latin"
    named.mkdir()

    assert resolve_artifact_dir(
        None, command="prune", model="org/draft", config=config, suffix="latin all"
    ) == tmp_path / "draft-latin-all"
    assert resolve_related_model_dir("target-latin") == named
    assert resolve_related_model_dir(str(named)) == named
    assert resolve_related_model_dir("org/target-latin") == tmp_path / "target-latin"


def test_related_path_bypasses_selection_and_accepts_tokenizerless_draft(monkeypatch, tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "old_to_new_token_ids.json").write_text(
        json.dumps({"0": 0, "2": 1, "3": 2}), encoding="utf-8"
    )
    (reference / "pruning_report.json").write_text(
        json.dumps({"original_vocab_size": 4, "source_model_vocab_size": 4}), encoding="utf-8"
    )
    output = tmp_path / "output"
    config = {
        "model_type": "draft",
        "architectures": ["DraftModel"],
        "vocab_size": 4,
        "dflash_config": {"mask_token_id": 3},
    }
    captured = {}

    monkeypatch.setattr(workflows, "load_model_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflows, "load_tokenizer", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(workflows, "run_token_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(workflows, "prune_gguf_checkpoint", lambda **kwargs: None)

    def prune(**kwargs):
        captured.update(kwargs)
        return PruneResult(output, 3, 1, {0: 0, 2: 1, 3: 2}, {0: 0, 1: 2, 2: 3})

    monkeypatch.setattr(workflows, "prune_safetensors_checkpoint", prune)
    monkeypatch.setattr(workflows, "write_prune_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        workflows, "save_related_tokenizer",
        lambda related, destination, count: captured.update(
            related_tokenizer=(related, destination, count)
        ),
    )
    monkeypatch.setattr(workflows, "write_checkpoint_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflows, "cached_checkpoint_status", lambda *args: type("Status", (), {"snapshot_path": None})())

    cli.main(
        [
            "prune",
            "draft",
            "--related-path",
            str(reference),
            "--save",
            str(output),
            "--workers",
            "1",
            "--yes",
        ]
    )

    assert captured["retained_token_ids"] == [0, 2, 3]
    assert captured["tokenizer"] is None
    assert captured["related_tokenizer"] == (reference, output, 3)
