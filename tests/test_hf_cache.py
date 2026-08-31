import os
import sys
import types
from pathlib import Path

from vocabpruner import hf_cache
from vocabpruner.paths import USER_ENV_TEMPLATE, hf_models_dir, load_user_env
from vocabpruner.pruning import resolve_checkpoint_source


def test_hf_home_selects_standard_hub_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))

    assert hf_models_dir() == tmp_path / "huggingface" / "hub"


def test_user_env_is_created_and_overrides_shell(monkeypatch, tmp_path):
    path = tmp_path / ".vocabpruner" / ".env"
    load_user_env(path, create=True)
    assert path.read_text(encoding="utf-8") == USER_ENV_TEMPLATE

    path.write_text("HF_HOME='/from-file'\nVOCABPRUNER_SAVE_DIR=/models\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", "/from-shell")
    monkeypatch.delenv("VOCABPRUNER_SAVE_DIR", raising=False)
    load_user_env(path)
    assert os.environ["HF_HOME"] == "/from-file"
    assert os.environ["VOCABPRUNER_SAVE_DIR"] == "/models"
    os.environ.pop("VOCABPRUNER_SAVE_DIR")


def test_analysis_download_marker_reports_incomplete_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_models"))

    hf_cache.mark_analysis_download("org/model", trust_remote_code=False)

    status = hf_cache.cached_checkpoint_status("org/model")
    assert status.complete is False
    assert status.kind == "unknown"
    assert (
        status.marker_path
        == tmp_path / "hf_models" / "models--org--model" / hf_cache.HF_DOWNLOAD_MARKER
    )
    assert status.snapshot_path is None


def test_checkpoint_marker_is_not_downgraded_by_later_analysis_load(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_models"))
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")

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
    (local_model / "model.safetensors").write_bytes(b"weights")

    status = hf_cache.cached_checkpoint_status(str(local_model))

    assert status.complete is True
    assert status.kind == "local"
    assert status.marker_path is None
    assert status.snapshot_path == local_model


def test_existing_analysis_snapshot_without_marker_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_models"))
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

    status = hf_cache.cached_checkpoint_status("org/model")

    assert status.complete is False
    assert status.kind == "unknown"
    assert status.snapshot_path is None


def test_existing_complete_checkpoint_is_detected_without_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_models"))
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"data")

    status = hf_cache.cached_checkpoint_status("org/model")

    assert status.complete is True
    assert status.kind == "checkpoint"
    assert status.snapshot_path == snapshot


def test_checkpoint_resolution_reuses_complete_marker_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_models"))
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
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
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_models"))
    snapshot = tmp_path / "hf_models" / "models--org--model" / "snapshots" / "abc123"
    calls = {}

    def fake_download(*args, **kwargs):
        calls.update(kwargs)
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"weights")
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
    assert marker["files"] == [{"path": "model.safetensors", "size": 7}]
