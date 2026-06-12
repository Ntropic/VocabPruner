from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from tokenpruner.models import TokenScore
from tokenpruner.vocab import TokenRecord, vocabulary_stats

ShowOption = Literal["all", "examples", "stats"]
ShowOptions = ShowOption | list[ShowOption] | tuple[ShowOption, ...] | set[ShowOption]


@dataclass(frozen=True)
class ExampleCounts:
    retained: int = 5
    pruned: int = 5


def render_jsonl(scores: list[TokenScore]) -> str:
    import json

    return "\n".join(json.dumps(score.to_dict(), ensure_ascii=False) for score in scores)


def render_markdown(
    model: str,
    records: list[TokenRecord],
    scores: list[TokenScore],
    seed: int,
    show: ShowOptions = ("all",),
) -> str:
    stats = vocabulary_stats(records)
    visible_scores = filter_scores(scores, show)
    retained = sorted(
        [score for score in visible_scores if score.status in {"protected", "retain_candidate"}],
        key=lambda score: score.token.token_id,
    )[:25]
    non_retained = sorted(
        [score for score in visible_scores if score.status == "non_retain_candidate"],
        key=lambda score: score.token.token_id,
    )[:25]
    stats_counts = score_stats(scores)

    parts = [
        "# TokenPruner Sample Report",
        "",
        f"- Model/tokenizer: `{model}`",
        f"- Vocabulary size: {stats['vocab_size']}",
        f"- Special tokens: {stats['special_tokens']}",
        f"- Byte fallback tokens: {stats['byte_fallback_tokens']}",
        f"- Classified tokens: {len(scores)}",
        f"- Displayed tokens: {len(visible_scores)}",
        f"- Show mode: {_show_label(show)}",
        f"- Random seed: {seed}",
        f"- Retained: {stats_counts['retained']}",
        f"- Pruned: {stats_counts['pruned']}",
        "",
        "## Retained Decisions",
        _table(retained),
        "",
        "## Pruned Decisions",
        _table(non_retained),
        "",
    ]
    return "\n".join(parts)


def render_plain_text(
    model: str,
    records: list[TokenRecord],
    scores: list[TokenScore],
    seed: int,
    show: ShowOptions = ("all",),
    example_counts: ExampleCounts = ExampleCounts(),
    style: Literal["plain", "rich"] = "plain",
) -> str:
    stats = vocabulary_stats(records)
    visible_scores = filter_scores(scores, show)
    sorted_scores = sorted(visible_scores, key=lambda score: score.token.token_id)
    stats_counts = score_stats(scores)

    parts = [
        "TokenPruner Sample Report",
        f"Model/tokenizer: {model}",
        f"Vocabulary size: {stats['vocab_size']}",
        f"Special tokens: {stats['special_tokens']}",
        f"Byte fallback tokens: {stats['byte_fallback_tokens']}",
        "",
        f"Classified tokens: {len(scores)}",
        f"Displayed tokens: {len(visible_scores)}",
        f"Show mode: {_show_label(show)}",
        f"Random seed: {seed}",
        f"Retained: {stats_counts['retained']}",
        f"Pruned: {stats_counts['pruned']}",
        "",
    ]
    if should_show_rows(show):
        parts.extend(["", "Decisions", _plain_rows(sorted_scores, style=style)])
    if should_show_examples(show):
        examples = example_groups(scores, example_counts, seed=seed)
        parts.extend(["", f"Retained examples ({len(examples['retained'])})", _plain_rows(examples["retained"], style=style)])
        parts.extend(["", f"Pruned examples ({len(examples['pruned'])})", _plain_rows(examples["pruned"], style=style)])
    return "\n".join(parts)


def filter_scores(scores: list[TokenScore], show: ShowOptions) -> list[TokenScore]:
    options = normalize_show_options(show)
    if "all" in options:
        return list(scores)
    return []


def should_show_rows(show: ShowOptions) -> bool:
    return "all" in normalize_show_options(show)


def should_show_examples(show: ShowOptions) -> bool:
    return "examples" in normalize_show_options(show)


def normalize_show_options(show: ShowOptions) -> set[ShowOption]:
    if isinstance(show, str):
        return {show}
    options = {option for option in show}
    unknown = options - {"all", "examples", "stats"}
    if unknown:
        raise ValueError(f"Unsupported show options: {', '.join(sorted(unknown))}")
    return options


def score_stats(scores: list[TokenScore]) -> dict[str, int]:
    counts = {"retained": 0, "pruned": 0}
    for score in scores:
        if score.status in {"protected", "retain_candidate"}:
            counts["retained"] += 1
        else:
            counts["pruned"] += 1
    return counts


def example_groups(scores: list[TokenScore], counts: ExampleCounts, seed: int = 13) -> dict[str, list[TokenScore]]:
    retained = [score for score in scores if score.status in {"protected", "retain_candidate"}]
    pruned = [score for score in scores if score.status not in {"protected", "retain_candidate"}]
    return {
        "retained": _seeded_examples(retained, counts.retained, seed, "retained"),
        "pruned": _seeded_examples(pruned, counts.pruned, seed, "pruned"),
    }


def _seeded_examples(scores: list[TokenScore], count: int, seed: int, bucket: str) -> list[TokenScore]:
    if count <= 0 or not scores:
        return []
    selected = list(scores)
    random.Random(f"{seed}:{bucket}").shuffle(selected)
    return selected[:count]


def _table(scores: list[TokenScore]) -> str:
    if not scores:
        return "_No samples._"
    rows = [
        "| id | raw | decoded | status | top explanation |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for score in scores:
        explanation = score.explanations[0] if score.explanations else ""
        rows.append(
            "| {id} | `{raw}` | `{decoded}` | {status} | {explanation} |".format(
                id=score.token.token_id,
                raw=_escape(score.token.raw_token),
                decoded=_escape(score.token.decoded_text),
                status=score.status,
                explanation=_escape(explanation),
            )
        )
    return "\n".join(rows)


def _plain_rows(scores: list[TokenScore], style: Literal["plain", "rich"] = "plain") -> str:
    if not scores:
        return "  No samples."
    lines = []
    for score in scores:
        scripts = ",".join(score.script_profile.scripts) or "-"
        decoded = repr(score.token.decoded_text)
        if style == "rich":
            decoded = f"[bold]{_rich_escape(decoded)}[/bold]"
        token_line = (
            "  {decoded}  id={id} status={status} scripts={scripts} raw={raw!r}"
        ).format(
            decoded=decoded,
            id=score.token.token_id,
            status=score.status,
            scripts=scripts,
            raw=score.token.raw_token,
        )
        lines.append(token_line)
        for explanation in score.explanations[:3]:
            lines.append(f"      - {explanation}")
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|").replace("\n", "\\n")


def _rich_escape(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]")


def _show_label(show: ShowOptions) -> str:
    options = normalize_show_options(show)
    return " ".join(
        option for option in ["stats", "all", "examples"] if option in options
    )
