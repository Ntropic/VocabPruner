from tokenpruner.models import ScriptProfile, TokenRecord, TokenScore
from tokenpruner.reports import (
    ExampleCounts,
    example_groups,
    filter_scores,
    render_plain_text,
    score_stats,
)


def make_score(token_id: int, status: str = "retain_candidate") -> TokenScore:
    record = TokenRecord(token_id, f"raw{token_id}", f"text{token_id}")
    return TokenScore(
        token=record,
        script_profile=ScriptProfile({"Latin": 5}, total_meaningful=5, low_confidence=False),
        protected=False,
        status=status,
        explanations=["test explanation"],
    )


def test_render_plain_text_includes_decisions_and_explanations():
    records = [TokenRecord(1, "hello", "hello")]
    scores = [
        TokenScore(
            token=records[0],
            script_profile=ScriptProfile({"Latin": 5}, total_meaningful=5, low_confidence=False),
            protected=False,
            status="retain_candidate",
            explanations=["decision tree: accept because every word segment was accepted"],
        )
    ]

    report = render_plain_text("tiny", records, scores, seed=42)

    assert "TokenPruner Sample Report" in report
    assert "Classified tokens: 1" in report
    assert "Displayed tokens: 1" in report
    assert "'hello'  id=1 status=retain_candidate" in report
    assert "      - decision tree: accept" in report


def test_render_plain_text_can_emit_rich_markup():
    records = [TokenRecord(1, "hello", "hello")]
    scores = [
        TokenScore(
            token=records[0],
            script_profile=ScriptProfile({"Latin": 5}, total_meaningful=5, low_confidence=False),
            protected=False,
            status="retain_candidate",
            explanations=["decision tree: accept because every word segment was accepted"],
        )
    ]

    report = render_plain_text("tiny", records, scores, seed=42, style="rich")

    assert "[bold]'hello'[/bold]" in report
    assert "status=retain_candidate" in report
    assert "score=" not in report


def test_filter_and_stats_are_binary():
    scores = [
        make_score(1),
        make_score(2, "non_retain_candidate"),
        make_score(3, "non_retain_candidate"),
        make_score(4, "non_retain_candidate"),
    ]

    assert [score.token.token_id for score in filter_scores(scores, ["all"])] == [1, 2, 3, 4]
    assert [score.token.token_id for score in filter_scores(scores, ["list"])] == [1, 2, 3, 4]
    assert filter_scores(scores, ["stats"]) == []
    assert score_stats(scores) == {"retained": 1, "pruned": 3}


def test_example_groups_samples_each_bucket_deterministically():
    scores = [
        make_score(1),
        make_score(2),
        make_score(7),
        make_score(3, "non_retain_candidate"),
        make_score(4, "non_retain_candidate"),
        make_score(8, "non_retain_candidate"),
    ]

    first = example_groups(scores, ExampleCounts(retained=2, pruned=2), seed=42)
    second = example_groups(scores, ExampleCounts(retained=2, pruned=2), seed=42)
    different_seed = example_groups(scores, ExampleCounts(retained=2, pruned=2), seed=43)

    assert first == second
    assert {score.token.token_id for score in first["retained"]} <= {1, 2, 7}
    assert {score.token.token_id for score in first["pruned"]} <= {3, 4, 8}
    assert first != different_seed


def test_render_plain_text_can_show_examples_only():
    records = [TokenRecord(i, f"raw{i}", f"text{i}") for i in range(1, 4)]
    scores = [
        make_score(1),
        make_score(2, "non_retain_candidate"),
        make_score(3, "non_retain_candidate"),
    ]

    report = render_plain_text(
        "tiny",
        records,
        scores,
        seed=13,
        show=["examples"],
        example_counts=ExampleCounts(retained=1, pruned=1),
    )

    assert "Retained examples (1)" in report
    assert "Pruned examples (1)" in report
    assert "'text1'  id=1 status=retain_candidate" in report
    assert "Decisions" not in report
