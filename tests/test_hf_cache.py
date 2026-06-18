import sys
import types
from pathlib import Path

from tokenpruner import hf_cache
from tokenpruner.pruning import resolve_checkpoint_source


def test_analysis_download_marker_reports_incomplete_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HF_MODELS_DIR", tmp_path / "hf_models")

    hf_cache.mark_analysis_download("org/model", trust_remote_code=False)

    status = hf_cache.cached_checkpoint_status("org/model")
    assert status.complete is False
    assert status.kind == "analysis"
    assert (
        status.marker_path
        == tmp_path / "hf_models" / "models--org--model" / hf_cache.HF_DOWNLOAD_MARKER
    )
    assert status.snapshot_path is None


def test_checkpoint_marker_is_not_downgraded_by_later_analysis_load(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HF_MODELS_DIR", tmp_path / "hf_models")
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"

    hf_cache.mark_checkpoint_download("org/model", trust_remote_code=True, snapshot_path=snapshot)
    hf_cache.mark_analysis_download("org/model", trust_remote_code=False)

    marker = hf_cache.read_download_marker("org/model")
    assert marker is not None
    assert marker["kind"] == "checkpoint"
    assert marker["snapshot_path"] == str(snapshot)

    status = hf_cache.cached_checkpoint_status("org/model")
    assert status.complete is True
    assert status.kind == "checkpoint"
    assert status.snapshot_path == snapshot


def test_local_model_path_is_complete_without_marker(tmp_path):
    local_model = tmp_path / "local-model"
    local_model.mkdir()

    status = hf_cache.cached_checkpoint_status(str(local_model))

    assert status.complete is True
    assert status.kind == "local"
    assert status.marker_path is None
    assert status.snapshot_path == local_model


def test_existing_analysis_snapshot_without_marker_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HF_MODELS_DIR", tmp_path / "hf_models")
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

    status = hf_cache.cached_checkpoint_status("org/model")

    assert status.complete is False
    assert status.kind == "unknown"
    assert status.snapshot_path is None


def test_existing_checkpoint_snapshot_without_marker_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HF_MODELS_DIR", tmp_path / "hf_models")
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"data")

    status = hf_cache.cached_checkpoint_status("org/model")

    assert status.complete is False
    assert status.kind == "unknown"
    assert status.snapshot_path is None


def test_checkpoint_resolution_reuses_complete_marker_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HF_MODELS_DIR", tmp_path / "hf_models")
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    hf_cache.mark_checkpoint_download("org/model", trust_remote_code=False, snapshot_path=snapshot)

    def fail_download(*args, **kwargs):
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fail_download),
    )

    assert resolve_checkpoint_source("org/model") == snapshot


def test_checkpoint_resolution_downloads_all_sidecars_but_ignores_unpruned_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HF_MODELS_DIR", tmp_path / "hf_models")
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    calls = {}

    def fake_download(*args, **kwargs):
        calls.update(kwargs)
        snapshot.mkdir(parents=True)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_download),
    )

    assert resolve_checkpoint_source("org/model") == snapshot
    assert calls["allow_patterns"] == ("*",)
    assert "*.safetensors" not in calls["ignore_patterns"]
    assert "*.bin" in calls["ignore_patterns"]

    marker = hf_cache.read_download_marker("org/model")
    assert marker is not None
    assert marker["allow_patterns"] == ["*"]
    assert "*.bin" in marker["ignore_patterns"]
