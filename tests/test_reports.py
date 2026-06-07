import json

from tokenpruner.models import ScriptProfile, TokenRecord, TokenScore
from tokenpruner.reports import (
    ExampleCounts,
    example_groups,
    filter_scores,
    render_plain_text,
    render_stats,
    score_stats,
    write_reports,
)


def make_score(token_id: int, retain: float, non_retain: float) -> TokenScore:
    record = TokenRecord(token_id, f"raw{token_id}", f"text{token_id}")
    return TokenScore(
        token=record,
        script_profile=ScriptProfile({"Latin": 5}, total_meaningful=5, low_confidence=False),
        retain_score=retain,
        non_retain_score=non_retain,
        symbol_score=0.0,
        dictionary_score=0.0,
        wordfreq_score=0.0,
        protected=False,
        status="retain_candidate",
        explanations=["test explanation"],
    )


def test_render_plain_text_includes_scores_and_explanations():
    records = [TokenRecord(1, "hello", "hello")]
    scores = [
        TokenScore(
            token=records[0],
            script_profile=ScriptProfile({"Latin": 5}, total_meaningful=5, low_confidence=False),
            retain_score=3.0,
            non_retain_score=0.0,
            symbol_score=0.0,
            dictionary_score=0.0,
            wordfreq_score=0.0,
            protected=False,
            status="retain_candidate",
            explanations=["Latin script contributes retained score 3.00"],
        )
    ]

    report = render_plain_text("tiny", records, scores, sample_size=1, seed=42)

    assert "TokenPruner Sample Report" in report
    assert "Sample size: 1" in report
    assert "Classified tokens: 1" in report
    assert "Displayed tokens: 1" in report
    assert "'hello'  score=3.00/0.00 id=1" in report
    assert "      - Latin script contributes retained score" in report


def test_render_plain_text_can_emit_rich_markup():
    records = [TokenRecord(1, "hello", "hello")]
    scores = [
        TokenScore(
            token=records[0],
            script_profile=ScriptProfile({"Latin": 5}, total_meaningful=5, low_confidence=False),
            retain_score=3.0,
            non_retain_score=1.0,
            symbol_score=0.0,
            dictionary_score=0.0,
            wordfreq_score=0.0,
            protected=False,
            status="retain_candidate",
            explanations=["Latin script contributes retained score 3.00"],
        )
    ]

    report = render_plain_text("tiny", records, scores, sample_size=1, seed=42, style="rich")

    assert "[bold]'hello'[/bold]" in report
    assert "score=[green]3.00[/green]/[red]1.00[/red]" in report


def test_ambiguous_filter_and_stats():
    scores = [
        make_score(1, 3.0, 0.0),
        make_score(2, 0.0, 4.0),
        make_score(3, 2.0, 1.0),
        make_score(4, 0.0, 0.0),
    ]

    assert [score.token.token_id for score in filter_scores(scores, ["ambiguous"])] == [3, 4]
    assert [score.token.token_id for score in filter_scores(scores, ["all"])] == [1, 2, 3, 4]
    assert [score.token.token_id for score in filter_scores(scores, ["json"])] == [1, 2, 3, 4]
    assert filter_scores(scores, ["stats"]) == []
    assert score_stats(scores) == {"retained": 1, "pruned": 1, "ambiguous": 2}


def test_render_stats_prints_counts_only():
    scores = [make_score(1, 3.0, 0.0), make_score(2, 0.0, 4.0), make_score(3, 0.0, 0.0)]

    report = render_stats("tiny", scores, sample_size=-1, seed=13)

    assert "TokenPruner Stats" in report
    assert "Sample size: all" in report
    assert "Classified: 3" in report
    assert "Retained: 1" in report
    assert "Pruned: 1" in report
    assert "Ambiguous: 1" in report
    assert "score=" not in report


def test_example_groups_samples_each_bucket_deterministically():
    scores = [
        make_score(1, 3.0, 0.0),
        make_score(2, 6.0, 0.0),
        make_score(7, 4.0, 0.0),
        make_score(3, 0.0, 4.0),
        make_score(4, 0.0, 7.0),
        make_score(8, 0.0, 5.0),
        make_score(5, 2.0, 1.9),
        make_score(6, 0.0, 0.0),
        make_score(9, 1.0, 1.0),
    ]

    first = example_groups(scores, ExampleCounts(retained=2, pruned=2, ambiguous=2), seed=42)
    second = example_groups(scores, ExampleCounts(retained=2, pruned=2, ambiguous=2), seed=42)
    different_seed = example_groups(scores, ExampleCounts(retained=2, pruned=2, ambiguous=2), seed=43)

    assert first == second
    assert {score.token.token_id for score in first["retained"]} <= {1, 2, 7}
    assert {score.token.token_id for score in first["pruned"]} <= {3, 4, 8}
    assert {score.token.token_id for score in first["ambiguous"]} <= {5, 6, 9}
    assert first != different_seed


def test_render_plain_text_can_show_examples_only():
    records = [TokenRecord(i, f"raw{i}", f"text{i}") for i in range(1, 4)]
    scores = [make_score(1, 3.0, 0.0), make_score(2, 0.0, 4.0), make_score(3, 0.0, 0.0)]

    report = render_plain_text(
        "tiny",
        records,
        scores,
        sample_size=-1,
        seed=13,
        show=["examples"],
        example_counts=ExampleCounts(retained=1, pruned=1, ambiguous=1),
    )

    assert "Retained examples (1)" in report
    assert "Pruned examples (1)" in report
    assert "Ambiguous examples (1)" in report
    assert "'text1'  score=3.00/0.00 id=1" in report
    assert "Scores" not in report


def test_write_reports_filters_ambiguous_rows(tmp_path):
    records = [TokenRecord(i, f"raw{i}", f"text{i}") for i in range(1, 5)]
    scores = [
        make_score(1, 3.0, 0.0),
        make_score(2, 0.0, 4.0),
        make_score(3, 2.0, 1.0),
        make_score(4, 0.0, 0.0),
    ]

    markdown_path, jsonl_path = write_reports(tmp_path, "tiny", records, scores, sample_size=-1, seed=13, show=["ambiguous"])
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    markdown = markdown_path.read_text(encoding="utf-8")

    assert [row["token"]["token_id"] for row in rows] == [3, 4]
    assert "Displayed tokens: 2" in markdown
    assert "`text3`" in markdown
    assert "`text1`" not in markdown
