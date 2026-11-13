import json

import pytest

from vocabpruner.errors import PruneRefusal
from vocabpruner.storage.metadata import write_prune_metadata
from vocabpruner.storage.metadata import save_related_tokenizer


def test_related_metadata_inherits_actual_selection(tmp_path):
    related = tmp_path / "related"
    related.mkdir()
    selection = {"keep_scripts": ["Latin"], "symbol_lists": ["emoji"]}
    (related / "vocabpruner_model.json").write_text(json.dumps({"score_config": selection}))
    output = write_prune_metadata(tmp_path / "draft", "draft-source", {}, related_model=related)
    metadata = json.loads(output.read_text())
    assert metadata["score_config"] == selection
    assert metadata["related_model"] == str(related.resolve())
    assert metadata["source_model"] == "draft-source"


def test_related_metadata_refuses_missing_provenance(tmp_path):
    with pytest.raises(PruneRefusal, match="Cannot inherit"):
        write_prune_metadata(tmp_path / "draft", "source", {}, related_model=tmp_path / "missing")


def test_related_tokenizer_refuses_mismatched_map(monkeypatch, tmp_path):
    class Tokenizer:
        def get_vocab(self):
            return {"hello": 0, "world": 2}

        def save_pretrained(self, output):
            raise AssertionError("Invalid tokenizer must not be saved")

    monkeypatch.setattr(
        "vocabpruner.tokenization.loading.load_tokenizer", lambda path: Tokenizer()
    )
    with pytest.raises(PruneRefusal, match="does not match"):
        save_related_tokenizer(tmp_path, tmp_path, 2)
