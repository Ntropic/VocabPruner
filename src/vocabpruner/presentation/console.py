from __future__ import annotations

from pathlib import Path
import sys

from vocabpruner.presentation.reports import (
    ExampleCounts,
    render_plain_text,
    should_show_examples,
    should_show_rows,
)


def print_pruning_preview(
    *,
    model: str,
    save: str,
    original_count: int,
    retained_count: int,
    pruned_count: int,
) -> None:
    percentage = (pruned_count / original_count * 100.0) if original_count else 0.0
    print("VocabPruner Pruning Preview")
    print(f"Model: {model}")
    print(f"Save: {save}")
    print(f"Original vocab size: {original_count}")
    print(f"Retained vocab size: {retained_count}")
    print(f"Pruned vocab size: {pruned_count} ({percentage:.2f}%)")


def print_cache_status(cache_dir: Path, reused: bool) -> None:
    status = "reused" if reused else "created"
    print(f"List {status}: {cache_dir}", file=sys.stderr)


def confirm_pruning() -> bool:
    try:
        answer = input("Continue with pruning? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def print_prune_decisions(
    *,
    model: str,
    records,
    scores,
    show: list[str],
    example_counts: ExampleCounts,
) -> None:
    if should_show_rows(show) or should_show_examples(show) or "stats" in set(show):
        report = render_plain_text(
            model,
            records,
            scores,
            seed=13,
            show=show,
            example_counts=example_counts,
            style="rich",
        )
        try:
            from rich.console import Console

            Console().print(report, soft_wrap=True)
        except ImportError:
            print(
                render_plain_text(
                    model,
                    records,
                    scores,
                    seed=13,
                    show=show,
                    example_counts=example_counts,
                )
            )


def normalize_show_args(
    show: list[str] | list[list[str]] | None,
    default: list[str] | None = None,
) -> tuple[list[str], ExampleCounts]:
    if not show:
        return list(default) if default is not None else ["all"], ExampleCounts()
    flattened: list[str] = []
    for item in show:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    normalized: list[str] = []
    example_counts = ExampleCounts()
    for item in flattened:
        if item == "examples" or item.startswith("examples:"):
            normalized.append("examples")
            example_counts = parse_examples_spec(item)
        else:
            normalized.append(item)
    valid = {"all", "examples", "stats"}
    unknown = [item for item in normalized if item not in valid]
    if unknown:
        raise SystemExit(f"Unsupported --show value(s): {', '.join(unknown)}")
    ordered = [
        item for item in ["stats", "all", "examples"] if item in normalized
    ]
    return ordered or ["all"], example_counts


def normalize_create_args(create: list[str] | list[list[str]] | None) -> list[str]:
    if not create:
        return []
    flattened: list[str] = []
    for item in create:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    valid = {"list", "decisions", "html"}
    unknown = [item for item in flattened if item not in valid]
    if unknown:
        raise SystemExit(f"Unsupported --create value(s): {', '.join(unknown)}")
    return [item for item in ["list", "decisions", "html"] if item in flattened]


def parse_examples_spec(value: str) -> ExampleCounts:
    if value == "examples":
        return ExampleCounts()
    prefix = "examples:"
    if not value.startswith(prefix):
        raise SystemExit(f"Invalid examples spec: {value}")
    spec = value[len(prefix) :]
    if not spec:
        raise SystemExit(f"Invalid examples spec: {value}")
    if spec.isdigit():
        count = int(spec)
        return ExampleCounts(retained=count, pruned=count)
    allowed = {"retained", "pruned"}
    counts = {"retained": 5, "pruned": 5}
    parts = spec.split(",")
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"Invalid examples spec: {value}")
        name, raw_count = part.split("=", 1)
        if name not in allowed:
            raise SystemExit(f"Invalid examples category: {name}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"Invalid examples count: {raw_count}") from exc
        if count < 0:
            raise SystemExit(f"Invalid examples count: {count}")
        counts[name] = count
    return ExampleCounts(**counts)
