import shutil
from pathlib import Path

import pytest

import tokenpruner.cli as cli
import tokenpruner.pipeline as pipeline
import tokenpruner.pruning as pruning


@pytest.fixture(autouse=True)
def allow_pruning_runtime(monkeypatch, tmp_path):
    lists_dir = tmp_path / "lists"
    decisions_dir = tmp_path / "decisions"
    pruned_models_dir = tmp_path / "pruned_models"
    monkeypatch.setattr(pipeline, "LISTS_DIR", lists_dir)
    monkeypatch.setattr(pipeline, "DECISIONS_DIR", decisions_dir)
    monkeypatch.setattr(pipeline, "PRUNED_MODELS_DIR", pruned_models_dir)
    monkeypatch.setattr(pruning, "missing_pruning_packages", lambda report: ())
    monkeypatch.setattr(pruning, "validate_pruning_runtime", lambda report: None)
    monkeypatch.setattr(cli, "prune_safetensors_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(
        cli,
        "start_html_server",
        lambda directory: ("http://127.0.0.1:8765/wordfreq_viewer.html", 12345),
    )
    yield
    shutil.rmtree(lists_dir, ignore_errors=True)
    shutil.rmtree(decisions_dir, ignore_errors=True)
    shutil.rmtree(pruned_models_dir, ignore_errors=True)


class TinyTokenizer:
    all_special_ids = [0]
    all_special_tokens = ["<s>"]
    special_tokens_map = {"bos_token": "<s>"}

    def get_vocab(self):
        return {"<s>": 0, "hello": 1, "Ж": 2, "{": 3, "": 4, "π": 5}

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        values = {0: "<s>", 1: "hello", 2: "Ж", 3: "{", 4: "", 5: "π"}
        return values[token_ids[0]]


class TinyConfig:
    model_type = "tiny"

    def to_dict(self):
        return {
            "model_type": "tiny",
            "architectures": ["TinyForCausalLM"],
            "vocab_size": 6,
            "tie_word_embeddings": True,
        }


def test_inspect_reports_model_architecture(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())

    cli.main(["inspect", "--model", "tiny"])

    output = capsys.readouterr().out
    assert "Tokenizer Vocabulary" in output
    assert "Model Architecture" in output
    assert "MTP: no" in output
    assert "root.vocab_size: 6" not in output
    assert "- vocab_size: 6" in output
    assert "- root: tied" in output


def test_pull_downloads_checkpoint(monkeypatch, capsys, tmp_path):
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(cli, "resolve_checkpoint_source", lambda *args, **kwargs: snapshot)

    cli.main(["pull", "--model", "tiny"])

    assert f"Checkpoint: {snapshot}" in capsys.readouterr().out


def test_analyze_output_print_accepts_space_separated_lists(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "analyze",
            "--model",
            "tiny",
            "--keep-script",
            "Latin",
            "Common",
            "Inherited",
            "--symbol-preset",
            "coding",
            "json",
            "--languages",
            "en",
            "--wordfreq-min-zipf",
            "3.5",
            "--seed",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Sample Report" in output
    assert "Classified tokens: 6" in output


def test_analyze_classifies_all(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--languages", "en", "--wordfreq-min-zipf", "4", "--show", "stats"])

    output = capsys.readouterr().out
    assert "Classified tokens: 6" in output


def test_analyze_rejects_sample_size_argument(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--sample-size", "0", "--show", "stats"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_show_stats_prints_stats_only(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "4", "--show", "stats"])

    output = capsys.readouterr().out
    assert "Retained:" in output
    assert "Pruned:" in output
    assert "Decisions" not in output
    assert "'hello'" not in output


def test_analyze_show_can_be_repeated(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "analyze",
            "--model",
            "tiny",
            "--wordfreq-min-zipf",
            "4",
            "--show",
            "stats",
            "--show",
            "examples:1",
        ]
    )

    output = capsys.readouterr().out
    assert "Show mode: stats examples" in output
    assert "Retained examples" in output


def test_analyze_show_all_prints_rows(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "4", "--show", "all"])

    output = capsys.readouterr().out
    assert "Decisions" in output
    assert "'hello'" in output


def test_analyze_create_list_creates_cache_without_decisions(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--seed", "1", "--create", "list"])

    captured = capsys.readouterr()
    assert "TokenPruner Sample Report" not in captured.out
    assert '"token"' not in captured.out
    assert "List:" in captured.err
    list_dirs = [path for path in pipeline.LISTS_DIR.iterdir() if path.is_dir()]
    assert len(list_dirs) == 1
    assert (list_dirs[0] / "settings.json").is_file()
    assert (list_dirs[0] / "token_facts.json").is_file()
    assert (list_dirs[0] / "wordfreq_facts.json").is_file()
    assert not list(pipeline.DECISIONS_DIR.rglob("token_decisions.jsonl"))


def test_analyze_create_list_reuses_existing_cache_without_threshold_decisions(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "8", "--create", "list"])
    first = capsys.readouterr()
    monkeypatch.setattr(
        cli,
        "load_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tokenizer should not load")),
    )
    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "3", "--create", "list"])
    second = capsys.readouterr()

    assert "List created" in first.err
    assert "List reused" in second.err
    assert "Retained:" not in first.out
    assert "Retained:" not in second.out
    assert not list(pipeline.DECISIONS_DIR.rglob("token_decisions.jsonl"))


def test_analyze_create_html_reuses_cache_without_loading_tokenizer(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--create", "list"])
    capsys.readouterr()
    monkeypatch.setattr(
        cli,
        "load_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tokenizer should not load")),
    )

    cli.main(["analyze", "--model", "tiny", "--create", "html"])

    captured = capsys.readouterr()
    assert "List reused" in captured.err
    assert "HTML:" in captured.out
    assert "Open: http://127.0.0.1:8765/wordfreq_viewer.html" in captured.out


def test_analyze_show_stats_requires_threshold(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--show", "stats"])
    except SystemExit as exc:
        assert "A word frequency threshold is required" in str(exc)
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_create_decisions_writes_jsonl_without_printing_it(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--seed", "1", "--wordfreq-min-zipf", "4", "--show", "stats", "--create", "decisions"])

    captured = capsys.readouterr()
    assert "TokenPruner Sample Report" in captured.out
    assert '"token"' not in captured.out
    assert "Decisions:" in captured.err
    saved_decisions = list(pipeline.DECISIONS_DIR.rglob("token_decisions.jsonl"))
    assert saved_decisions
    assert len(saved_decisions[0].read_text(encoding="utf-8").strip().splitlines()) == 6


def test_analyze_accepts_workers(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--workers", "1", "--wordfreq-min-zipf", "4", "--show", "stats"])

    output = capsys.readouterr().out
    assert "Classified tokens: 6" in output


def test_normalize_show_examples_default_counts():
    show, counts = cli.normalize_show_args([["examples"]])

    assert show == ["examples"]
    assert counts.retained == 5
    assert counts.pruned == 5


def test_normalize_show_examples_inline_count():
    show, counts = cli.normalize_show_args([["examples:3"]])

    assert show == ["examples"]
    assert counts.retained == 3
    assert counts.pruned == 3


def test_normalize_show_examples_category_counts():
    show, counts = cli.normalize_show_args([["stats", "examples:retained=2,pruned=4"]])

    assert show == ["stats", "examples"]
    assert counts.retained == 2
    assert counts.pruned == 4


def test_normalize_show_rejects_invalid_examples_specs():
    cases = {
        "examples:abc": "Invalid examples spec: examples:abc",
        "examples:retained=-1": "Invalid examples count: -1",
        "examples:unknown=3": "Invalid examples category: unknown",
    }
    for value, message in cases.items():
        try:
            cli.normalize_show_args([[value]])
        except SystemExit as exc:
            assert str(exc) == message
        else:
            raise AssertionError(f"Expected SystemExit for {value}")


def test_analyze_show_examples_prints_decision_buckets(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "4", "--show", "examples:2"])

    output = capsys.readouterr().out
    assert "Show mode: examples" in output
    assert "Retained examples" in output
    assert "Pruned examples" in output
    assert "Decisions" not in output


def test_analyze_show_stats_examples_prints_both(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "4", "--show", "stats", "examples:2"])

    output = capsys.readouterr().out
    assert "Retained:" in output
    assert "Retained examples" in output
    assert "Show mode: stats examples" in output


def test_analyze_help_does_not_list_example_count(capsys):
    try:
        cli.main(["analyze", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected SystemExit")

    output = capsys.readouterr().out
    assert "--example-count" not in output
    assert "examples:retained=20,pruned=20" in output


def test_analyze_rejects_save_argument(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "4", "--save", "reports"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_create_decisions_uses_generated_settings_path(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "analyze",
            "--model",
            "tiny",
            "--seed",
            "1",
            "--wordfreq-min-zipf",
            "4",
            "--show",
            "stats",
            "--create",
            "decisions",
        ]
    )

    captured = capsys.readouterr()
    expected = (
        pipeline.DECISIONS_DIR
        / "tiny__scripts-common-inherited-latin__symbols-ascii-punctuation-coding__langs-en__zipf-4__seed-1"
        / "token_decisions.jsonl"
    )
    assert '"token"' not in captured.out
    assert "Decisions:" in captured.err
    assert expected.exists()
    assert len(expected.read_text(encoding="utf-8").strip().splitlines()) == 6


def test_analyze_show_json_is_rejected(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--show", "json"])
    except SystemExit as exc:
        assert str(exc) == "Unsupported --show value(s): json"
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_show_list_is_rejected(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--show", "list"])
    except SystemExit as exc:
        assert str(exc) == "Unsupported --show value(s): list"
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_create_html_writes_viewer_assets(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--create", "html"])

    captured = capsys.readouterr()
    assert "HTML:" in captured.out
    assert "Open: http://127.0.0.1:8765/wordfreq_viewer.html" in captured.out
    assert "Server PID: 12345" in captured.err
    assert "TokenPruner Sample Report" not in captured.out
    html_paths = list(pipeline.LISTS_DIR.rglob("wordfreq_viewer.html"))
    assert html_paths
    list_dir = html_paths[0].parent
    assert (list_dir / "settings.json").is_file()
    assert (list_dir / "token_facts.json").is_file()
    assert (list_dir / "wordfreq_facts.json").is_file()
    assert not (list_dir / "wordfreq_data.js").exists()
    html = html_paths[0].read_text(encoding="utf-8")
    assert 'fetch("settings.json")' in html
    assert 'fetch("token_facts.json")' in html
    assert 'fetch("wordfreq_facts.json")' in html
    assert 'id="stats"' in html
    assert "metric.id = 'metric'" in html
    assert "token words plotted" in html
    assert 'token "' in html
    assert "hoverPoint" in html
    assert "threshold" not in html.lower()


def test_analyze_show_html_is_rejected(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--show", "html"])
    except SystemExit as exc:
        assert str(exc) == "Unsupported --show value(s): html"
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_threshold_change_reuses_same_cache(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "8", "--show", "stats"])
    first = capsys.readouterr()
    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "3", "--show", "stats"])
    second = capsys.readouterr()

    assert "List created" in first.err
    assert "List reused" in second.err
    list_dirs = [path for path in pipeline.LISTS_DIR.iterdir() if path.is_dir()]
    assert len(list_dirs) == 1
    assert "zipf-8" not in list_dirs[0].name
    assert "zipf-3" not in list_dirs[0].name


def test_analyze_without_save_writes_cache_only(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-min-zipf", "4", "--show", "stats"])

    capsys.readouterr()
    assert pipeline.LISTS_DIR.exists()
    assert any((path / "settings.json").is_file() for path in pipeline.LISTS_DIR.iterdir() if path.is_dir())
    assert not list(tmp_path.rglob("retained_token_ids.txt"))
    assert not list(tmp_path.rglob("pruned_token_ids.txt"))


def test_create_list_uses_default_cache_path(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny/model", "--languages", "en", "--create", "list"])

    captured = capsys.readouterr()
    expected = (
        pipeline.LISTS_DIR
        / "tiny-model__scripts-common-inherited-latin__symbols-ascii-punctuation-coding__langs-en"
    )
    assert '"token"' not in captured.out
    assert "List:" in captured.err
    assert (expected / "settings.json").exists()
    assert (expected / "token_facts.json").exists()
    assert (expected / "wordfreq_facts.json").exists()
    assert not list(pipeline.DECISIONS_DIR.rglob("token_decisions.jsonl"))


def test_prune_refuses_separate_smaller_vocabulary_before_loading_weights(monkeypatch, tmp_path):
    class MultiVocabConfig:
        model_type = "tiny_multi_vocab"

        def to_dict(self):
            return {
                "model_type": "tiny_multi_vocab",
                "vocab_size": 6,
                "talker_config": {"vocab_size": 4},
            }

    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: MultiVocabConfig())

    def fail_load(*args, **kwargs):
        raise AssertionError("weights should not be loaded for refused architectures")

    monkeypatch.setattr(cli, "load_causal_lm", fail_load)

    try:
        cli.main(
            [
                "prune",
                "--model",
                "tiny",
                "--wordfreq-min-zipf",
                "4",
                "--save",
                str(tmp_path / "out"),
            ]
        )
    except SystemExit as exc:
        assert "separate token mapping" in str(exc)
    else:
        raise AssertionError("Expected SystemExit")


def test_prune_confirmation_decline_avoids_loading_weights(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    def fail_load(*args, **kwargs):
        raise AssertionError("weights should not be loaded after declining")

    monkeypatch.setattr(cli, "load_causal_lm", fail_load)

    cli.main(
        [
            "prune",
            "--model",
            "tiny",
            "--wordfreq-min-zipf",
            "4",
            "--save",
            str(tmp_path / "out"),
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Pruning Preview" in output
    assert "Pruned vocab size:" in output
    assert "Pruning cancelled." in output


def test_prune_without_save_path_uses_default_artifact_path(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(
        cli,
        "load_causal_lm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("weights should not load")),
    )

    cli.main(
        [
            "prune",
            "--model",
            "tiny/model",
            "--wordfreq-min-zipf",
            "4",
        ]
    )

    output = capsys.readouterr().out
    expected = (
        pipeline.PRUNED_MODELS_DIR
        /  "tiny-model__scripts-common-inherited-latin__symbols-ascii-punctuation-coding__langs-en__zipf-4"
    )
    assert f"Save: {expected}" in output
    assert "Pruning cancelled." in output


def test_prune_defaults_to_summary_and_can_show_examples(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    def fail_load(*args, **kwargs):
        raise AssertionError("weights should not be loaded after declining")

    monkeypatch.setattr(cli, "load_causal_lm", fail_load)

    cli.main(
        [
            "prune",
            "--model",
            "tiny",
            "--wordfreq-min-zipf",
            "4",
            "--save",
            str(tmp_path / "out"),
            "--show",
            "examples:1",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Pruning Preview" in output
    assert "TokenPruner Sample Report" in output
    assert "Show mode: examples" in output
    assert "Retained examples" in output
    assert "Pruned examples" in output
    assert "Pruning cancelled." in output


def test_prune_default_show_is_stats(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(
        cli,
        "load_causal_lm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("weights should not load")),
    )

    cli.main(
        [
            "prune",
            "--model",
            "tiny",
            "--wordfreq-min-zipf",
            "4",
            "--save",
            str(tmp_path / "out"),
        ]
    )

    output = capsys.readouterr().out
    assert "Show mode: stats" in output
    assert "Retained examples" not in output


def test_prune_yes_skips_confirmation(monkeypatch, capsys, tmp_path):
    from tokenpruner.pruning import PruneResult

    output_dir = tmp_path / "out"
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())
    monkeypatch.setattr(cli, "load_causal_lm", lambda *args, **kwargs: object())
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError("input should not run")))
    monkeypatch.setattr(
        cli,
        "prune_model_and_tokenizer",
        lambda **kwargs: PruneResult(
            output_dir,
            3,
            3,
            {0: 0, 1: 1, 3: 2},
            {0: 0, 1: 1, 2: 3},
            original_checkpoint_bytes=4 * 1024 ** 3,
            pruned_checkpoint_bytes=3 * 1024 ** 3,
        ),
    )

    cli.main(
        [
            "prune",
            "--model",
            "tiny",
            "--wordfreq-min-zipf",
            "4",
            "--save",
            str(output_dir),
            "--yes",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Pruning Preview" in output
    assert "TokenPruner Prune" in output
    assert "Checkpoint size: 4.000 GB -> 3.000 GB (1.000 GB, 25.00% smaller)" in output


def test_prune_uses_streaming_backend_without_loading_full_model(monkeypatch, capsys, tmp_path):
    from tokenpruner.pruning import PruneResult

    output_dir = tmp_path / "streamed"
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(cli, "load_model_config", lambda *args, **kwargs: TinyConfig())
    monkeypatch.setattr(
        cli,
        "load_causal_lm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full model should not load")),
    )

    def fake_stream(**kwargs):
        assert kwargs["output_dir"] == str(output_dir)
        return PruneResult(
            output_dir,
            3,
            3,
            {0: 0, 1: 1, 3: 2},
            {0: 0, 1: 1, 2: 3},
            original_checkpoint_bytes=2 * 1024 ** 3,
            pruned_checkpoint_bytes=1024 ** 3,
        )

    monkeypatch.setattr(cli, "prune_safetensors_checkpoint", fake_stream)

    cli.main(
        [
            "prune",
            "--model",
            "tiny",
            "--wordfreq-min-zipf",
            "4",
            "--save",
            str(output_dir),
            "--yes",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Prune" in output
    assert f"Save: {output_dir}" in output
    assert "Checkpoint size: 2.000 GB -> 1.000 GB (1.000 GB, 50.00% smaller)" in output


def test_confirm_pruning_accepts_only_explicit_yes(monkeypatch):
    for answer in ("y", "Y", "yes", " YES "):
        monkeypatch.setattr("builtins.input", lambda prompt, answer=answer: answer)
        assert cli.confirm_pruning() is True
    for answer in ("", "n", "no", "maybe"):
        monkeypatch.setattr("builtins.input", lambda prompt, answer=answer: answer)
        assert cli.confirm_pruning() is False


def test_ensure_pruning_runtime_offers_and_installs_missing_packages(monkeypatch, capsys):
    calls = []
    checks = iter([("torch", "nvidia-modelopt"), ()])
    monkeypatch.setattr(pruning, "missing_pruning_packages", lambda report: next(checks))
    monkeypatch.setattr(pruning, "validate_pruning_runtime", lambda report: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    monkeypatch.setattr(pruning.subprocess, "run", lambda command, check: calls.append((command, check)))

    pruning.ensure_pruning_runtime(object())

    output = capsys.readouterr().out
    assert "Missing packages" in output
    assert "python" in output.lower()
    assert calls == [([pruning.sys.executable, "-m", "pip", "install", "torch", "nvidia-modelopt"], True)]


def test_ensure_pruning_runtime_decline_does_not_install(monkeypatch):
    monkeypatch.setattr(pruning, "missing_pruning_packages", lambda report: ("torch",))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(
        pruning.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pip should not run")),
    )

    with pytest.raises(SystemExit, match="were not installed"):
        pruning.ensure_pruning_runtime(object())


def test_ensure_pruning_runtime_yes_skips_install_prompt(monkeypatch):
    calls = []
    monkeypatch.setattr(pruning, "missing_pruning_packages", lambda report: ("compressed-tensors",))
    monkeypatch.setattr(pruning, "validate_pruning_runtime", lambda report: None)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (_ for _ in ()).throw(AssertionError("input should not run")),
    )
    monkeypatch.setattr(pruning.subprocess, "run", lambda command, check: calls.append(command))

    pruning.ensure_pruning_runtime(object(), assume_yes=True)

    assert calls == [[pruning.sys.executable, "-m", "pip", "install", "compressed-tensors"]]


def test_analyze_rejects_output_argument(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--output", "decisions.jsonl"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit")


def test_rejects_both_wordfreq_threshold_modes(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(
            [
                "analyze",
                "--model",
                "tiny",
                "--wordfreq-min-zipf",
                "4",
                "--wordfreq-top-pct",
                "1",
                "--show",
                "stats",
            ]
        )
    except SystemExit as exc:
        assert str(exc) == "Use either --wordfreq-min-zipf or --wordfreq-top-pct, not both."
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_rejects_drop_script_argument(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--drop-script", "Han"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_rejects_unknown_show_value(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--show", "nope"])
    except SystemExit as exc:
        assert str(exc) == "Unsupported --show value(s): nope"
    else:
        raise AssertionError("Expected SystemExit")
