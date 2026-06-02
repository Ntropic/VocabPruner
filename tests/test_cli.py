from pathlib import Path

import tokenpruner.cli as cli


class TinyTokenizer:
    all_special_ids = [0]
    all_special_tokens = ["<s>"]
    special_tokens_map = {"bos_token": "<s>"}

    def get_vocab(self):
        return {"<s>": 0, "hello": 1, "Ж": 2, "{": 3, "": 4}

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        values = {0: "<s>", 1: "hello", 2: "Ж", 3: "{", 4: ""}
        return values[token_ids[0]]


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


def test_analyze_default_sample_size_scores_all(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--wordfreq-lang", "en", "--show", "stats"])

    output = capsys.readouterr().out
    assert "Sample size: all" in output
    assert "Classified tokens: 5" in output


def test_analyze_rejects_invalid_sample_size(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--sample-size", "0", "--show", "stats"])
    except SystemExit as exc:
        assert str(exc) == "--sample-size must be -1 or at least 1"
    else:
        raise AssertionError("Expected SystemExit")


def test_analyze_show_ambiguous_filters_print_rows(monkeypatch, capsys, tmp_path):
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
            "--wordfreq-lang",
            "en",
            "--sample-size",
            "-1",
            "--show",
            "ambiguous",
        ]
    )

    output = capsys.readouterr().out
    assert "Displayed tokens:" in output
    assert "''" in output
    assert "'hello'" not in output


def test_analyze_show_stats_prints_stats_only(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "stats"])

    output = capsys.readouterr().out
    assert "Retained:" in output
    assert "Pruned:" in output
    assert "Ambiguous:" in output
    assert "Scores" not in output
    assert "'hello'" not in output


def test_analyze_show_stats_ambiguous_prints_stats_and_ambiguous_rows(monkeypatch, capsys, tmp_path):
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
            "ambiguous",
        ]
    )

    output = capsys.readouterr().out
    assert "Retained:" in output
    assert "Scores" in output
    assert "''" in output
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
            "ambiguous",
        ]
    )

    output = capsys.readouterr().out
    assert "Show mode: stats ambiguous" in output
    assert "Scores" in output
    assert "''" in output
    assert "'hello'" not in output


def test_analyze_show_ambiguous_words_prints_decoded_and_raw_only(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "ambiguous-words"])

    output = capsys.readouterr().out
    assert "Ambiguous words" in output
    assert "'' / ''" in output
    assert "score=" not in output
    assert "Scores" not in output


def test_analyze_show_all_prints_non_ambiguous_rows(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "-1", "--show", "all"])

    output = capsys.readouterr().out
    assert "Scores" in output
    assert "'hello'" in output


def test_analyze_show_json_outputs_jsonl(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    cli.main(["analyze", "--model", "tiny", "--sample-size", "2", "--seed", "1", "--show", "json"])

    output = capsys.readouterr().out
    assert "TokenPruner Sample Report" not in output
    assert '"token"' in output
    assert len(output.strip().splitlines()) == 2


def test_analyze_save_writes_jsonl_file(monkeypatch, capsys, tmp_path):
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
            "scores.jsonl",
        ]
    )

    output = capsys.readouterr().out
    saved = Path("scores.jsonl").read_text(encoding="utf-8")
    assert "Retained:" in output
    assert '"token"' not in output
    assert len(saved.strip().splitlines()) == 2


def test_analyze_save_with_json_writes_visible_rows(monkeypatch, capsys, tmp_path):
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
            "nested/scores.jsonl",
        ]
    )

    output = capsys.readouterr().out
    saved = Path("nested/scores.jsonl").read_text(encoding="utf-8")
    assert '"token"' in output
    assert len(saved.strip().splitlines()) == 2


def test_analyze_save_with_ambiguous_saves_ambiguous_json_rows(monkeypatch, capsys, tmp_path):
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
            "ambiguous",
            "--save",
            "ambiguous.jsonl",
        ]
    )

    capsys.readouterr()
    saved = Path("ambiguous.jsonl").read_text(encoding="utf-8")
    assert "'hello'" not in saved
    assert '"decoded_text": ""' in saved


def test_analyze_rejects_output_argument(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: TinyTokenizer())

    try:
        cli.main(["analyze", "--model", "tiny", "--sample-size", "1", "--output", "print"])
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
