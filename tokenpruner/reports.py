from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tokenpruner.models import TokenScore
from tokenpruner.vocab import TokenRecord, vocabulary_stats

ShowOption = Literal["all", "ambiguous", "ambiguous-words", "examples", "stats", "json"]
ShowOptions = ShowOption | list[ShowOption] | tuple[ShowOption, ...] | set[ShowOption]


@dataclass(frozen=True)
class ExampleCounts:
    retained: int = 5
    pruned: int = 5
    ambiguous: int = 5


def write_reports(
    output_dir: str | Path,
    model: str,
    records: list[TokenRecord],
    scores: list[TokenScore],
    sample_size: int,
    seed: int,
    show: ShowOptions = ("all",),
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "sampled_token_scores.jsonl"
    markdown_path = output / "summary.md"
    visible_scores = filter_scores(scores, show)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for score in visible_scores:
            handle.write(json.dumps(score.to_dict(), ensure_ascii=False) + "\n")

    markdown_path.write_text(
        render_markdown(model, records, scores, sample_size, seed, show=show),
        encoding="utf-8",
    )
    return markdown_path, jsonl_path


def render_jsonl(scores: list[TokenScore]) -> str:
    return "\n".join(json.dumps(score.to_dict(), ensure_ascii=False) for score in scores)


def render_markdown(
    model: str,
    records: list[TokenRecord],
    scores: list[TokenScore],
    sample_size: int,
    seed: int,
    show: ShowOptions = ("all",),
) -> str:
    stats = vocabulary_stats(records)
    visible_scores = filter_scores(scores, show)
    retained = sorted(visible_scores, key=lambda score: score.retain_score, reverse=True)[:25]
    non_retained = sorted(visible_scores, key=lambda score: score.non_retain_score, reverse=True)[:25]
    ambiguous = sorted(
        [score for score in visible_scores if is_ambiguous_score(score)],
        key=lambda score: abs(score.retain_score - score.non_retain_score),
    )[:25]
    stats_counts = score_stats(scores)

    parts = [
        "# TokenPruner Sample Report",
        "",
        f"- Model/tokenizer: `{model}`",
        f"- Vocabulary size: {stats['vocab_size']}",
        f"- Special tokens: {stats['special_tokens']}",
        f"- Byte fallback tokens: {stats['byte_fallback_tokens']}",
        f"- Sample size: {_sample_size_label(sample_size)}",
        f"- Classified tokens: {len(scores)}",
        f"- Displayed tokens: {len(visible_scores)}",
        f"- Show mode: {_show_label(show)}",
        f"- Random seed: {seed}",
        f"- Retained: {stats_counts['retained']}",
        f"- Pruned: {stats_counts['pruned']}",
        f"- Ambiguous: {stats_counts['ambiguous']}",
        "",
        "## Highest Retained Scores",
        _table(retained),
        "",
        "## Highest Non-Retained Scores",
        _table(non_retained),
        "",
        "## Ambiguous Samples",
        _table(ambiguous),
        "",
    ]
    return "\n".join(parts)


def render_plain_text(
    model: str,
    records: list[TokenRecord],
    scores: list[TokenScore],
    sample_size: int,
    seed: int,
    show: ShowOptions = ("all",),
    example_counts: ExampleCounts = ExampleCounts(),
    style: Literal["plain", "rich"] = "plain",
) -> str:
    stats = vocabulary_stats(records)
    visible_scores = filter_scores(scores, show)
    sorted_scores = sorted(visible_scores, key=lambda score: score.retain_score, reverse=True)
    stats_counts = score_stats(scores)

    parts = [
        "TokenPruner Sample Report",
        f"Model/tokenizer: {model}",
        f"Vocabulary size: {stats['vocab_size']}",
        f"Special tokens: {stats['special_tokens']}",
        f"Byte fallback tokens: {stats['byte_fallback_tokens']}",
        "",
        f"Sample size: {_sample_size_label(sample_size)}",
        f"Classified tokens: {len(scores)}",
        f"Displayed tokens: {len(visible_scores)}",
        f"Show mode: {_show_label(show)}",
        f"Random seed: {seed}",
        f"Retained: {stats_counts['retained']}",
        f"Pruned: {stats_counts['pruned']}",
        f"Ambiguous: {stats_counts['ambiguous']}",
        "",
    ]
    if should_show_rows(show):
        parts.extend(["", "Scores", _plain_rows(sorted_scores, style=style)])
    if should_show_ambiguous_words(show):
        parts.extend(["", "Ambiguous words", _ambiguous_word_rows(filter_scores(scores, ["ambiguous"]))])
    if should_show_examples(show):
        examples = example_groups(scores, example_counts, seed=seed)
        parts.extend(["", f"Retained examples ({len(examples['retained'])})", _plain_rows(examples["retained"], style=style)])
        parts.extend(["", f"Pruned examples ({len(examples['pruned'])})", _plain_rows(examples["pruned"], style=style)])
        parts.extend(
            ["", f"Ambiguous examples ({len(examples['ambiguous'])})", _plain_rows(examples["ambiguous"], style=style)]
        )
    return "\n".join(parts)


def render_stats(model: str, scores: list[TokenScore], sample_size: int, seed: int) -> str:
    stats_counts = score_stats(scores)
    return "\n".join(
        [
            "TokenPruner Stats",
            f"Model/tokenizer: {model}",
            f"Sample size: {_sample_size_label(sample_size)}",
            f"Classified: {len(scores)}",
            f"Random seed: {seed}",
            f"Retained: {stats_counts['retained']}",
            f"Pruned: {stats_counts['pruned']}",
            f"Ambiguous: {stats_counts['ambiguous']}",
        ]
    )


def filter_scores(scores: list[TokenScore], show: ShowOptions) -> list[TokenScore]:
    options = normalize_show_options(show)
    if "ambiguous" in options:
        return [score for score in scores if is_ambiguous_score(score)]
    if "all" in options or "json" in options:
        return list(scores)
    return []


def should_show_rows(show: ShowOptions) -> bool:
    options = normalize_show_options(show)
    return "all" in options or "ambiguous" in options


def should_show_json(show: ShowOptions) -> bool:
    return "json" in normalize_show_options(show)


def should_show_ambiguous_words(show: ShowOptions) -> bool:
    return "ambiguous-words" in normalize_show_options(show)


def should_show_examples(show: ShowOptions) -> bool:
    return "examples" in normalize_show_options(show)


def normalize_show_options(show: ShowOptions) -> set[ShowOption]:
    if isinstance(show, str):
        return {show}
    options = {option for option in show}
    unknown = options - {"all", "ambiguous", "ambiguous-words", "examples", "stats", "json"}
    if unknown:
        raise ValueError(f"Unsupported show options: {', '.join(sorted(unknown))}")
    return options


def score_stats(scores: list[TokenScore]) -> dict[str, int]:
    counts = {"retained": 0, "pruned": 0, "ambiguous": 0}
    for score in scores:
        if is_ambiguous_score(score):
            counts["ambiguous"] += 1
        elif score.retain_score > score.non_retain_score:
            counts["retained"] += 1
        else:
            counts["pruned"] += 1
    return counts


def is_ambiguous_score(score: TokenScore) -> bool:
    return (score.retain_score > 0 and score.non_retain_score > 0) or (
        score.retain_score == 0 and score.non_retain_score == 0
    )


def example_groups(scores: list[TokenScore], counts: ExampleCounts, seed: int = 13) -> dict[str, list[TokenScore]]:
    retained = [
        score for score in scores if not is_ambiguous_score(score) and score.retain_score > score.non_retain_score
    ]
    pruned = [
        score for score in scores if not is_ambiguous_score(score) and score.non_retain_score >= score.retain_score
    ]
    ambiguous = [score for score in scores if is_ambiguous_score(score)]
    return {
        "retained": _seeded_examples(retained, counts.retained, seed, "retained"),
        "pruned": _seeded_examples(pruned, counts.pruned, seed, "pruned"),
        "ambiguous": _seeded_examples(ambiguous, counts.ambiguous, seed, "ambiguous"),
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
        "| id | raw | decoded | retain | non-retain | status | top explanation |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for score in scores:
        explanation = score.explanations[0] if score.explanations else ""
        rows.append(
            "| {id} | `{raw}` | `{decoded}` | {retain:.2f} | {drop:.2f} | {status} | {explanation} |".format(
                id=score.token.token_id,
                raw=_escape(score.token.raw_token),
                decoded=_escape(score.token.decoded_text),
                retain=score.retain_score,
                drop=score.non_retain_score,
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
        retain = f"{score.retain_score:.2f}"
        non_retain = f"{score.non_retain_score:.2f}"
        if style == "rich":
            decoded = f"[bold]{_rich_escape(decoded)}[/bold]"
            retain = f"[green]{retain}[/green]"
            non_retain = f"[red]{non_retain}[/red]"
        token_line = (
            "  {decoded}  score={retain}/{drop} "
            "id={id} status={status} scripts={scripts} raw={raw!r}"
        ).format(
            decoded=decoded,
            retain=retain,
            drop=non_retain,
            id=score.token.token_id,
            status=score.status,
            scripts=scripts,
            raw=score.token.raw_token,
        )
        lines.append(token_line)
        for explanation in score.explanations[:3]:
            lines.append(f"      - {explanation}")
    return "\n".join(lines)


def _ambiguous_word_rows(scores: list[TokenScore]) -> str:
    if not scores:
        return "  No samples."
    return "\n".join(f"  {score.token.decoded_text!r} / {score.token.raw_token!r}" for score in scores)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|").replace("\n", "\\n")


def _rich_escape(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]")


def _sample_size_label(sample_size: int) -> str:
    return "all" if sample_size == -1 else str(sample_size)


def _show_label(show: ShowOptions) -> str:
    options = normalize_show_options(show)
    return " ".join(
        option for option in ["stats", "all", "ambiguous", "ambiguous-words", "examples", "json"] if option in options
    )
