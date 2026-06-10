from pathlib import Path

import pytest

import tokenpruner.cli as cli


@pytest.fixture(autouse=True)
def allow_pruning_runtime(monkeypatch):
    monkeypatch.setattr(cli, "missing_pruning_packages", lambda report: ())
    monkeypatch.setattr(cli, "validate_pruning_runtime", lambda report: None)
    monkeypatch.setattr(cli, "prune_safetensors_checkpoint", lambda **kwargs: None)


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
            "--wordfreq-lang",
            "en",
            "--wordfreq-min-zipf",
            "3.5",
            "--sample-size",
            "2",
            "--seed",
            "1",
            "--workers",
            "2",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Sample Report" in output
    assert "Classified tokens: 2" in output


def test_default_workers_is_cpu_count_minus_two(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 12)

    assert cli.default_workers() == 10


def test_default_workers_has_minimum_one(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 2)

    assert cli.default_workers() == 1


def test_analyze_default_sample_size_classifies_all(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-lang", "en", "--show", "stats"])

    output = capsys.readouterr().out
    assert "Sample size: all" in output
    assert "Classified tokens: 6" in output


def test_analyze_rejects_invalid_sample_size(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--sample-size", "0", "--show", "stats"])
    except SystemExit as exc:
        assert str(exc) == "--sample-size must be -1 or at least 1"
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_show_stats_prints_stats_only(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "stats"])

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
            "--sample-size",
            "-1",
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

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "all"])

    output = capsys.readouterr().out
    assert "Decisions" in output
    assert "'hello'" in output


def test_analyze_show_json_outputs_jsonl(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "2", "--seed", "1", "--show", "json"])

    output = capsys.readouterr().out
    assert "TokenPruner Sample Report" not in output
    assert '"token"' in output
    assert len(output.strip().splitlines()) == 2


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

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "examples:2"])

    output = capsys.readouterr().out
    assert "Show mode: examples" in output
    assert "Retained examples" in output
    assert "Pruned examples" in output
    assert "Decisions" not in output


def test_analyze_show_stats_examples_prints_both(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "stats", "examples:2"])

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


def test_analyze_save_writes_report_artifacts(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "analyze",
            "--model",
            "tiny",
            "--sample-size",
            "2",
            "--seed",
            "1",
            "--show",
            "stats",
            "--save",
            "reports",
        ]
    )

    output = capsys.readouterr().out
    saved = Path("reports/token_decisions.jsonl").read_text(encoding="utf-8")
    assert "Retained:" in output
    assert len(saved.strip().splitlines()) == 2
    assert Path("reports/summary.md").exists()
    assert Path("reports/policy.json").exists()


def test_analyze_bare_save_uses_default_artifact_path(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "analyze",
            "--model",
            "tiny/model",
            "--sample-size",
            "2",
            "--seed",
            "1",
            "--show",
            "stats",
            "--save",
        ]
    )

    capsys.readouterr()
    expected = (
        Path("tokenpruner_artifacts")
        / "analyze"
        / "tiny-model__scripts-common-inherited-latin__symbols-ascii-punctuation-coding__langs-en__zipf-4__sample-2__seed-1"
    )
    assert (tmp_path / expected / "token_decisions.jsonl").exists()
    assert (tmp_path / expected / "summary.md").exists()
    assert (tmp_path / expected / "policy.json").exists()


def test_analyze_show_json_and_save_are_independent(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "analyze",
            "--model",
            "tiny",
            "--sample-size",
            "2",
            "--seed",
            "1",
            "--show",
            "json",
            "--save",
            "nested/reports",
        ]
    )

    output = capsys.readouterr().out
    saved = Path("nested/reports/token_decisions.jsonl").read_text(encoding="utf-8")
    assert '"token"' in output
    assert len(saved.strip().splitlines()) == 2


def test_analyze_without_save_writes_no_files(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "1", "--workers", "1", "--show", "stats"])

    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []


def test_list_command_writes_token_artifacts(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(
        [
            "list",
            "--model",
            "tiny",
            "--save",
            "lists",
            "--wordfreq-lang",
            "en",
            "--workers",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Token Lists" in output
    assert Path("lists/retained_token_ids.txt").exists()
    assert Path("lists/pruned_token_ids.txt").exists()
    assert Path("lists/token_decisions.jsonl").exists()
    assert Path("lists/policy.json").exists()


def test_list_without_save_path_uses_default_artifact_path(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["list", "--model", "tiny/model", "--wordfreq-lang", "en", "--workers", "1"])

    output = capsys.readouterr().out
    expected = (
        Path("tokenpruner_artifacts")
        / "list"
        / "tiny-model__scripts-common-inherited-latin__symbols-ascii-punctuation-coding__langs-en__zipf-4"
    )
    assert f"Save: {expected}" in output
    assert (tmp_path / expected / "retained_token_ids.txt").exists()
    assert (tmp_path / expected / "pruned_token_ids.txt").exists()


def test_prune_refuses_separate_smaller_vocabulary_before_loading_weights(monkeypatch, tmp_path):
    class MultiVocabConfig:
        model_type = "tiny_multi_vocab"

        def to_dict(self):
            return {
                "model_type": "tiny_multi_vocab",
                "vocab_size": 6,
                "talker_config": {"vocab_size": 4},
            }

    retained = tmp_path / "retained_token_ids.txt"
    retained.write_text("0\n1\n", encoding="utf-8")
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
                "--save",
                str(tmp_path / "out"),
                "--retained-token-ids",
                str(retained),
                "--workers",
                "1",
            ]
        )
    except SystemExit as exc:
        assert "separate token mapping" in str(exc)
    else:
        raise AssertionError("Expected SystemExit")


def test_prune_confirmation_decline_avoids_loading_weights(monkeypatch, capsys, tmp_path):
    retained = tmp_path / "retained_token_ids.txt"
    retained.write_text("0\n1\n3\n", encoding="utf-8")
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
            "--save",
            str(tmp_path / "out"),
            "--retained-token-ids",
            str(retained),
            "--workers",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "TokenPruner Pruning Preview" in output
    assert "Pruned vocab size: 3 (50.00%)" in output
    assert "Pruning cancelled." in output


def test_prune_without_save_path_uses_default_artifact_path(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    retained = tmp_path / "retained_token_ids.txt"
    retained.write_text("0\n1\n3\n", encoding="utf-8")
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
            "--retained-token-ids",
            str(retained),
            "--workers",
            "1",
        ]
    )

    output = capsys.readouterr().out
    expected = (
        Path("tokenpruner_artifacts")
        / "prune"
        / "tiny-model__scripts-common-inherited-latin__symbols-ascii-punctuation-coding__langs-en__zipf-4__ids-retained-token-ids"
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
            "--save",
            str(tmp_path / "out"),
            "--workers",
            "1",
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
            "--save",
            str(tmp_path / "out"),
            "--workers",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "Show mode: stats" in output
    assert "Retained examples" not in output


def test_prune_retained_list_rejects_example_output(monkeypatch, tmp_path):
    retained = tmp_path / "retained_token_ids.txt"
    retained.write_text("0\n1\n3\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(
            [
                "prune",
                "--model",
                "tiny",
                "--save",
                str(tmp_path / "out"),
                "--retained-token-ids",
                str(retained),
                "--show",
                "examples",
                "--workers",
                "1",
            ]
        )
    except SystemExit as exc:
        assert "requires full classification" in str(exc)
    else:
        raise AssertionError("Expected SystemExit")


def test_prune_yes_skips_confirmation(monkeypatch, capsys, tmp_path):
    from tokenpruner.pruning import PruneResult

    retained = tmp_path / "retained_token_ids.txt"
    retained.write_text("0\n1\n3\n", encoding="utf-8")
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
            "--save",
            str(output_dir),
            "--retained-token-ids",
            str(retained),
            "--workers",
            "1",
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
            "--save",
            str(output_dir),
            "--workers",
            "1",
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
    monkeypatch.setattr(cli, "missing_pruning_packages", lambda report: next(checks))
    monkeypatch.setattr(cli, "validate_pruning_runtime", lambda report: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append((command, check)))

    cli.ensure_pruning_runtime(object())

    output = capsys.readouterr().out
    assert "Missing packages" in output
    assert "python" in output.lower()
    assert calls == [([cli.sys.executable, "-m", "pip", "install", "torch", "nvidia-modelopt"], True)]


def test_ensure_pruning_runtime_decline_does_not_install(monkeypatch):
    monkeypatch.setattr(cli, "missing_pruning_packages", lambda report: ("torch",))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pip should not run")),
    )

    with pytest.raises(SystemExit, match="were not installed"):
        cli.ensure_pruning_runtime(object())


def test_ensure_pruning_runtime_yes_skips_install_prompt(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "missing_pruning_packages", lambda report: ("compressed-tensors",))
    monkeypatch.setattr(cli, "validate_pruning_runtime", lambda report: None)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (_ for _ in ()).throw(AssertionError("input should not run")),
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append(command))

    cli.ensure_pruning_runtime(object(), assume_yes=True)

    assert calls == [[cli.sys.executable, "-m", "pip", "install", "compressed-tensors"]]


def test_analyze_rejects_output_argument(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--sample-size", "1", "--output", "decisions.jsonl"])
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
        cli.main(["analyze", "--model", "tiny", "--sample-size", "1", "--show", "nope"])
    except SystemExit as exc:
        assert str(exc) == "Unsupported --show value(s): nope"
    else:
        raise AssertionError("Expected SystemExit")
