from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from tokenpruner.architecture import inspect_model_architecture, load_model_config, render_model_architecture
from tokenpruner.metadata import write_prune_metadata
from tokenpruner.pipeline import (
    cached_create_only_pipeline,
    resolve_artifact_dir,
    run_token_pipeline,
    write_create_artifacts,
)
from tokenpruner.pruning import (
    PruneRefusal,
    ensure_pruning_runtime,
    load_causal_lm,
    model_vocabulary_size,
    prune_model_and_tokenizer,
    prune_safetensors_checkpoint,
    resolve_checkpoint_source,
    validate_pruning_architecture,
)
from tokenpruner.reports import (
    ExampleCounts,
    format_gb,
    render_plain_text,
    should_show_examples,
    should_show_rows,
)
from tokenpruner.scoring import require_wordfreq_threshold, settings_config
from tokenpruner.hf_cache import cached_checkpoint_status
from tokenpruner.paths import PRUNED_MODELS_DIR
from tokenpruner.token_lists import (
    compact_token_maps,
)
from tokenpruner.vocab import load_tokenizer, tokenizer_records, vocabulary_stats

DEFAULT_SAVE = "__tokenpruner_default_save__"


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenpruner",
        description="Tokenizer vocabulary analysis and cache-backed pruning validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect tokenizer vocabulary metadata.")
    inspect_parser.add_argument("--model", required=True, help="Hugging Face model ID or local tokenizer path.")
    inspect_parser.add_argument("--trust-remote-code", action="store_true")

    pull_parser = subparsers.add_parser("pull", help="Download checkpoint shards into the TokenPruner HF cache.")
    pull_parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path.")
    pull_parser.add_argument("--trust-remote-code", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Classify tokenizer tokens through the shared cached pipeline.")
    analyze_parser.add_argument("--model", required=True, help="Hugging Face model ID or local tokenizer path.")
    add_settings_arguments(analyze_parser)
    analyze_parser.add_argument("--seed", type=int, default=13)
    add_worker_argument(analyze_parser)
    add_show_argument(analyze_parser)
    add_create_argument(analyze_parser)
    analyze_parser.add_argument("--trust-remote-code", action="store_true")

    prune_parser = subparsers.add_parser(
        "prune",
        help="Prune supported language or multimodal models with compact text-token IDs.",
    )
    prune_parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path.")
    add_save_argument(prune_parser, "Path for the pruned model.", default=DEFAULT_SAVE)
    prune_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the pruning confirmation prompt.",
    )
    add_settings_arguments(prune_parser)
    add_worker_argument(prune_parser)
    add_show_argument(prune_parser)
    add_create_argument(prune_parser)
    prune_parser.add_argument("--trust-remote-code", action="store_true")

    return parser


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keep-script", nargs="+", default=["Latin", "Common", "Inherited"])
    parser.add_argument("--symbol-preset", nargs="+", default=["coding", "ascii-punctuation"])
    parser.add_argument("--dictionary", action="append", default=[], help="Custom dictionary text file.")
    parser.add_argument("--languages", nargs="+", default=["en"], help="Languages for word frequency scoring.")
    parser.add_argument("--wordfreq-min-zipf", type=float)
    parser.add_argument("--wordfreq-top-pct", type=float)


def add_worker_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Process workers for token fact creation. Defaults to CPU count minus 2.",
    )


def add_save_argument(parser: argparse.ArgumentParser, help_text: str, default: str | None = None) -> None:
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_SAVE,
        default=default,
        metavar="PATH",
        help=f"{help_text} Omit PATH to use the generated default path.",
    )


def add_show_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--show",
        nargs="+",
        action="append",
        metavar="SHOW",
        default=None,
        help=(
            "What to print: stats, all, examples. "
            "Can be combined. Examples accepts examples, examples:10, or "
            "examples:retained=20,pruned=20."
        ),
    )


def add_create_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--create",
        nargs="+",
        action="append",
        metavar="ARTIFACT",
        default=None,
        help="Artifacts to create without printing them. Supported: list, decisions, html.",
    )


def inspect_command(model: str, trust_remote_code: bool = False) -> None:
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    config = load_model_config(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    stats = vocabulary_stats(records)

    print("Tokenizer Vocabulary")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print()
    print(render_model_architecture(inspect_model_architecture(config)))


def pull_command(model: str, trust_remote_code: bool = False) -> None:
    source = resolve_checkpoint_source(model, trust_remote_code=trust_remote_code)
    print(f"Checkpoint: {source}")


def analyze_command(
    model: str,
    keep_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    languages: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    seed: int,
    workers: int,
    show: list[str],
    create: list[str] | list[list[str]] | None,
    trust_remote_code: bool = False,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    raw_show = show
    create = normalize_create_args(create)
    show, example_counts = normalize_show_args(raw_show, default=[] if create else None)
    derive_decisions = bool(raw_show) or "decisions" in create or not create
    if derive_decisions:
        require_wordfreq_threshold(wordfreq_min_zipf, wordfreq_top_pct)
    config = settings_config(
        keep_script,
        symbol_preset,
        dictionary,
        languages,
        wordfreq_min_zipf,
        wordfreq_top_pct,
    )
    result = cached_create_only_pipeline(model, config) if create and not derive_decisions else None
    if result is None:
        tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
        records = tokenizer_records(tokenizer)
        result = run_token_pipeline(
            model=model,
            config=config,
            records=records,
            workers=workers,
            derive_decisions=derive_decisions,
            progress=True,
        )
    print_cache_status(result.cache_dir, result.cache_reused)
    records = result.records
    scores = result.scores
    if create:
        decisions_dir = resolve_artifact_dir(
            None,
            command="analyze",
            model=model,
            config=config,
            seed=seed,
        )
        write_create_artifacts(
            create,
            decisions_dir,
            config=config,
            scores=scores,
            cache_dir=result.cache_dir,
            html_dir=result.cache_dir,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
            html_launcher=start_html_server,
        )
    if show and (should_show_rows(show) or should_show_examples(show) or "stats" in set(show)):
        report = render_plain_text(
            model, records, scores, seed, show=show, example_counts=example_counts, style="rich"
        )
        try:
            from rich.console import Console

            Console().print(report, soft_wrap=True)
        except ImportError:
            print(render_plain_text(model, records, scores, seed, show=show, example_counts=example_counts))

def prune_command(
    model: str,
    save: str,
    keep_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    languages: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    workers: int,
    show: list[str] | list[list[str]] | None = None,
    create: list[str] | list[list[str]] | None = None,
    yes: bool = False,
    trust_remote_code: bool = False,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    show, example_counts = normalize_show_args(show, default=["stats"])
    create = normalize_create_args(create)
    require_wordfreq_threshold(wordfreq_min_zipf, wordfreq_top_pct)
    model_config = load_model_config(model, trust_remote_code=trust_remote_code)
    architecture_report = inspect_model_architecture(model_config)
    try:
        validate_pruning_architecture(architecture_report, len(records))
    except PruneRefusal as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint_vocab_size = model_vocabulary_size(architecture_report, len(records))
    config = settings_config(
        keep_script,
        symbol_preset,
        dictionary,
        languages,
        wordfreq_min_zipf,
        wordfreq_top_pct,
    )
    save_dir = resolve_artifact_dir(
        None if save == DEFAULT_SAVE else save,
        command="prune",
        model=model,
        config=config,
    )
    save = str(save_dir)
    result = run_token_pipeline(
        model=model,
        config=config,
        records=records,
        workers=workers,
        progress=True,
    )
    print_cache_status(result.cache_dir, result.cache_reused)
    scores = result.scores
    retained_ids = result.retained_ids
    write_create_artifacts(
        create,
        save_dir,
        config=config,
        scores=scores,
        cache_dir=result.cache_dir,
        html_dir=result.cache_dir,
        wordfreq_min_zipf=wordfreq_min_zipf,
        wordfreq_top_pct=wordfreq_top_pct,
        html_launcher=start_html_server,
    )

    try:
        try:
            compact_token_maps(retained_ids)
        except ValueError as exc:
            raise PruneRefusal(str(exc)) from exc
        if not retained_ids:
            raise PruneRefusal("Refusing to prune to an empty vocabulary.")
        if min(retained_ids) < 0 or max(retained_ids) >= len(records):
            raise PruneRefusal("Retained token IDs must be within the tokenizer vocabulary.")
        retained_count = len(retained_ids)
        pruned_count = len(records) - retained_count
        print_pruning_preview(
            model=model,
            save=save,
            original_count=len(records),
            retained_count=retained_count,
            pruned_count=pruned_count,
        )
        print_prune_decisions(
            model=model,
            records=records,
            scores=scores,
            show=show,
            example_counts=example_counts,
        )
        if not yes and not _is_model_downloaded(model):
            print("Model checkpoint is not cached locally; pruning may download it.")
        if not yes and not confirm_pruning():
            print("Pruning cancelled.")
            return
        result = prune_safetensors_checkpoint(
            model_source=model,
            tokenizer=tokenizer,
            config=model_config,
            output_dir=save,
            retained_token_ids=retained_ids,
            original_vocab_size=len(records),
            model_vocab_size=checkpoint_vocab_size,
            trust_remote_code=trust_remote_code,
        )
        if result is None:
            ensure_pruning_runtime(architecture_report, assume_yes=yes)
            loaded_model = load_causal_lm(
                model,
                trust_remote_code=trust_remote_code,
                architecture_report=architecture_report,
            )
            source_dir = cached_checkpoint_status(model).snapshot_path
            result = prune_model_and_tokenizer(
                model=loaded_model,
                tokenizer=tokenizer,
                output_dir=save,
                retained_token_ids=retained_ids,
                original_vocab_size=len(records),
                model_vocab_size=checkpoint_vocab_size,
                source_dir=source_dir if source_dir is not None and source_dir.is_dir() else None,
            )
    except PruneRefusal as exc:
        raise SystemExit(str(exc)) from exc

    print("TokenPruner Prune")
    print(f"Model: {model}")
    print(f"Save: {result.output_dir}")
    print(f"Original vocab size: {len(records)}")
    print(f"Retained vocab size: {result.retained_count}")
    print(f"Pruned vocab size: {result.pruned_count}")
    if result.original_checkpoint_bytes is not None and result.pruned_checkpoint_bytes is not None:
        reduction = result.original_checkpoint_bytes - result.pruned_checkpoint_bytes
        percentage = (reduction / result.original_checkpoint_bytes * 100.0) if result.original_checkpoint_bytes else 0.0
        print(
            "Checkpoint size: "
            f"{format_gb(result.original_checkpoint_bytes)} GB -> "
            f"{format_gb(result.pruned_checkpoint_bytes)} GB "
            f"({format_gb(reduction)} GB, {percentage:.2f}% smaller)"
        )
    elif result.pruned_checkpoint_bytes is not None:
        print(f"Pruned checkpoint size: {format_gb(result.pruned_checkpoint_bytes)} GB")
    print("Transformed vocabulary surfaces:")
    for surface in result.transformed_surfaces:
        shared = f", shared with {surface.shared_with}" if surface.shared_with else ""
        quantized = ", quantized" if surface.quantized else ""
        print(f"- {surface.name}: {surface.kind}{shared}{quantized}")
    metadata_source = cached_checkpoint_status(model).snapshot_path
    write_prune_metadata(
        save,
        source_model=str(metadata_source if metadata_source is not None and metadata_source.exists() else model),
        score_config=config,
    )


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        inspect_command(args.model, trust_remote_code=args.trust_remote_code)
    elif args.command == "pull":
        pull_command(args.model, trust_remote_code=args.trust_remote_code)
    elif args.command == "analyze":
        analyze_command(
            model=args.model,
            keep_script=args.keep_script,
            symbol_preset=args.symbol_preset,
            dictionary=args.dictionary,
            languages=args.languages,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            seed=args.seed,
            workers=args.workers,
            show=args.show,
            create=args.create,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.command == "prune":
        prune_command(
            model=args.model,
            save=args.save,
            keep_script=args.keep_script,
            symbol_preset=args.symbol_preset,
            dictionary=args.dictionary,
            languages=args.languages,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            workers=args.workers,
            show=args.show,
            create=args.create,
            yes=args.yes,
            trust_remote_code=args.trust_remote_code,
        )


def app() -> None:
    main()


def start_html_server(directory: Path) -> tuple[str, int]:
    port = free_local_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
        ],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"http://127.0.0.1:{port}/wordfreq_viewer.html", int(process.pid)


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def print_pruning_preview(
    *,
    model: str,
    save: str,
    original_count: int,
    retained_count: int,
    pruned_count: int,
) -> None:
    percentage = (pruned_count / original_count * 100.0) if original_count else 0.0
    print("TokenPruner Pruning Preview")
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


def _is_model_downloaded(model: str) -> bool:
    return cached_checkpoint_status(model).complete


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


if __name__ == "__main__":
    main()
