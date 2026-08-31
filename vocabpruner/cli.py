from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from vocabpruner.architecture import inspect_model_architecture, load_model_config, render_model_architecture
from vocabpruner.codebooks import codebook_config_ids
from vocabpruner.metadata import write_prune_metadata
from vocabpruner.pipeline import (
    cached_create_only_pipeline,
    resolve_artifact_dir,
    run_token_pipeline,
    write_create_artifacts,
)
from vocabpruner.paths import load_user_env, pruned_models_dir
from vocabpruner.pruning import (
    PruneRefusal,
    ensure_pruning_runtime,
    load_causal_lm,
    model_vocabulary_size,
    prune_model_and_tokenizer,
    prune_gguf_checkpoint,
    prune_safetensors_checkpoint,
    resolve_checkpoint_source,
    validate_pruning_architecture,
)
from vocabpruner.reports import (
    ExampleCounts,
    format_gib,
    render_plain_text,
    should_show_examples,
    should_show_rows,
)
from vocabpruner.scoring import settings_config
from vocabpruner.scripts import script_lists
from vocabpruner.symbols import symbol_lists, syntax_lists
from vocabpruner.hf_cache import cached_checkpoint_status, write_checkpoint_manifest
from vocabpruner.gguf import resolve_gguf
from vocabpruner.groups import selection_groups
from vocabpruner.token_lists import (
    compact_token_maps,
)
from vocabpruner.vocab import load_tokenizer, tokenizer_records, vocabulary_stats

DEFAULT_SAVE = "__vocabpruner_default_save__"


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


def load_related_token_ids(directory: Path, source_vocab_size: int) -> list[int]:
    map_path = directory / "old_to_new_token_ids.json"
    report_path = directory / "pruning_report.json"
    if not map_path.is_file() or not report_path.is_file():
        raise PruneRefusal(f"Related model directory lacks pruning map metadata: {directory}")
    try:
        raw_map = json.loads(map_path.read_text(encoding="utf-8"))
        old_to_new = {int(old): int(new) for old, new in raw_map.items()}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reference_vocab_size = int(report.get("source_model_vocab_size", report["original_vocab_size"]))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PruneRefusal(f"Invalid related pruning map metadata: {directory}") from exc
    if "source_model_vocab_size" not in report and reference_vocab_size != source_vocab_size:
        try:
            metadata = json.loads((directory / "vocabpruner_model.json").read_text(encoding="utf-8"))
            source_dir = Path(metadata["source_model"]).expanduser()
            source_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
            source_sizes = {size for _, size in inspect_model_architecture(source_config).vocabulary_surfaces}
            if source_vocab_size in source_sizes:
                reference_vocab_size = source_vocab_size
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
    expected, _ = compact_token_maps(list(old_to_new))
    if not old_to_new or old_to_new != expected:
        raise PruneRefusal("Related token map must be a nonempty compact old-ID to new-ID map.")
    if reference_vocab_size != source_vocab_size:
        raise PruneRefusal(
            f"Related source vocabulary has {reference_vocab_size} tokens; model has {source_vocab_size}."
        )
    if min(old_to_new) < 0 or max(old_to_new) >= source_vocab_size:
        raise PruneRefusal("Related token map contains IDs outside the source vocabulary.")
    return sorted(old_to_new)


def resolve_related_model_dir(model_or_path: str) -> Path:
    path = Path(model_or_path).expanduser()
    if path.exists() or path.is_absolute() or model_or_path.startswith((".", "~")):
        return path.resolve()
    name = path.name
    return pruned_models_dir() / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vocabpruner",
        description="Tokenizer vocabulary analysis and cache-backed pruning validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the optional ~/.vocabpruner/.env file.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect tokenizer vocabulary metadata.")
    inspect_parser.add_argument("model", metavar="MODEL", help="Hugging Face model ID or local tokenizer path.")
    inspect_parser.add_argument("--trust-remote-code", action="store_true")

    pull_parser = subparsers.add_parser("pull", help="Download checkpoint shards into the Hugging Face cache.")
    pull_parser.add_argument("model", metavar="MODEL", help="Hugging Face model ID or local model path.")
    pull_parser.add_argument("--trust-remote-code", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Classify tokenizer tokens through the shared cached pipeline.")
    analyze_parser.add_argument("model", metavar="MODEL", help="Hugging Face model ID or local tokenizer path.")
    add_settings_arguments(analyze_parser)
    analyze_parser.add_argument("--seed", type=int, default=13)
    add_worker_argument(analyze_parser)
    add_show_argument(analyze_parser)
    add_create_argument(analyze_parser)
    analyze_parser.add_argument("--recreate-list", action="store_true", help="Rebuild the cached token list.")
    analyze_parser.add_argument("--trust-remote-code", action="store_true")

    prune_parser = subparsers.add_parser(
        "prune",
        help="Prune supported language or multimodal models with compact text-token IDs.",
    )
    prune_parser.add_argument("model", metavar="MODEL", help="Hugging Face model ID or local model path.")
    save_group = prune_parser.add_mutually_exclusive_group()
    add_save_argument(save_group, "Path for the pruned model.", default=DEFAULT_SAVE)
    save_group.add_argument(
        "--save-suffix",
        type=save_suffix_arg,
        metavar="SUFFIX",
        help="Append SUFFIX to the generated default pruned-model name.",
    )
    prune_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the pruning confirmation prompt.",
    )
    add_settings_arguments(prune_parser)
    add_worker_argument(prune_parser)
    add_show_argument(prune_parser)
    add_create_argument(prune_parser)
    prune_parser.add_argument("--recreate-list", action="store_true", help="Rebuild the cached token list.")
    prune_parser.add_argument("--trust-remote-code", action="store_true")
    related = prune_parser.add_mutually_exclusive_group()
    related.add_argument("--related-path", metavar="PATH", help="Reuse a token map from a pruned model directory.")
    related.add_argument(
        "--related-model",
        metavar="MODEL",
        help="Reuse a token map from an existing path or a model in the default pruned-model directory.",
    )

    return parser


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--selection-group",
        nargs="+",
        choices=selection_groups(),
        default=[],
        help="Apply file-defined script, syntax, and symbol selections.",
    )
    parser.add_argument(
        "--keep-script",
        nargs="+",
        choices=[*script_lists(), "all"],
        default=["Latin", "Common", "Inherited"],
    )
    parser.add_argument(
        "--syntax-list",
        nargs="+",
        choices=[*syntax_lists(), "all"],
        default=[],
        help="Syntax lists to preserve.",
    )
    parser.add_argument(
        "--symbol-list",
        nargs="+",
        choices=[*symbol_lists(), "all"],
        default=[],
        help="Symbol lists to preserve.",
    )
    parser.add_argument("--dictionary", action="append", default=[], help="Custom dictionary text file.")
    parser.add_argument("--languages", nargs="+", default=["en"], help="Languages for word frequency scoring.")
    parser.add_argument("--wordfreq-min-zipf", type=float)
    parser.add_argument("--wordfreq-top-pct", type=float)


def save_suffix_arg(value: str) -> str:
    if not any(character.isascii() and character.isalnum() for character in value):
        raise argparse.ArgumentTypeError("suffix must contain an ASCII letter or number")
    return value


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
    syntax_list: list[str],
    symbol_list: list[str],
    dictionary: list[str],
    languages: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    seed: int,
    workers: int,
    show: list[str],
    create: list[str] | list[list[str]] | None,
    recreate_list: bool = False,
    trust_remote_code: bool = False,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    raw_show = show
    create = normalize_create_args(create)
    show, example_counts = normalize_show_args(raw_show, default=[] if create else None)
    derive_decisions = bool(raw_show) or "decisions" in create or not create
    config = settings_config(
        keep_script,
        syntax_list,
        symbol_list,
        dictionary,
        languages,
        wordfreq_min_zipf,
        wordfreq_top_pct,
    )
    result = cached_create_only_pipeline(model, config) if create and not derive_decisions and not recreate_list else None
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
            recreate_list=recreate_list,
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
    syntax_list: list[str],
    symbol_list: list[str],
    dictionary: list[str],
    languages: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    workers: int,
    show: list[str] | list[list[str]] | None = None,
    create: list[str] | list[list[str]] | None = None,
    recreate_list: bool = False,
    yes: bool = False,
    trust_remote_code: bool = False,
    related_path: str | None = None,
    related_model: str | None = None,
    save_suffix: str | None = None,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    show, example_counts = normalize_show_args(show, default=["stats"])
    create = normalize_create_args(create)
    model_config = load_model_config(model, trust_remote_code=trust_remote_code)
    architecture_report = inspect_model_architecture(model_config)
    config = settings_config(
        keep_script,
        syntax_list,
        symbol_list,
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
        suffix=save_suffix,
    )
    save = str(save_dir)
    related_dir = Path(related_path).expanduser().resolve() if related_path else None
    if related_model:
        related_dir = resolve_related_model_dir(related_model)

    try:
        required_ids = codebook_config_ids(model_config)
        if related_dir is not None:
            if not architecture_report.vocabulary_surfaces:
                raise PruneRefusal("Related-map pruning requires a declared model vocabulary size.")
            checkpoint_vocab_size = architecture_report.vocabulary_surfaces[0][1]
            validate_pruning_architecture(architecture_report, checkpoint_vocab_size)
            retained_ids = load_related_token_ids(related_dir, checkpoint_vocab_size)
            tokenizer = None
            original_vocab_size = checkpoint_vocab_size
        else:
            tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
            records = tokenizer_records(tokenizer)
            validate_pruning_architecture(architecture_report, len(records))
            checkpoint_vocab_size = model_vocabulary_size(architecture_report, len(records))
            result = run_token_pipeline(
                model=model,
                config=config,
                records=records,
                workers=workers,
                progress=True,
                recreate_list=recreate_list,
            )
            print_cache_status(result.cache_dir, result.cache_reused)
            scores = result.scores
            retained_ids = sorted(set(result.retained_ids) | required_ids)
            original_vocab_size = len(records)
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

        missing_config_ids = required_ids - set(retained_ids)
        if missing_config_ids:
            raise PruneRefusal(
                "Related token map omits required config token ID(s): "
                + ", ".join(map(str, sorted(missing_config_ids)))
            )

        try:
            compact_token_maps(retained_ids)
        except ValueError as exc:
            raise PruneRefusal(str(exc)) from exc
        if not retained_ids:
            raise PruneRefusal("Refusing to prune to an empty vocabulary.")
        if min(retained_ids) < 0 or max(retained_ids) >= original_vocab_size:
            raise PruneRefusal("Retained token IDs must be within the source vocabulary.")
        retained_count = len(retained_ids)
        pruned_count = original_vocab_size - retained_count
        print_pruning_preview(
            model=model,
            save=save,
            original_count=original_vocab_size,
            retained_count=retained_count,
            pruned_count=pruned_count,
        )
        if related_dir is None:
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
        result = prune_gguf_checkpoint(
            model_source=model,
            output_dir=save,
            retained_token_ids=retained_ids,
            original_vocab_size=original_vocab_size,
        )
        if result is None:
            result = prune_safetensors_checkpoint(
                model_source=model,
                tokenizer=tokenizer,
                config=model_config,
                output_dir=save,
                retained_token_ids=retained_ids,
                original_vocab_size=original_vocab_size,
                model_vocab_size=checkpoint_vocab_size,
                trust_remote_code=trust_remote_code,
            )
        if result is None:
            if tokenizer is None:
                raise PruneRefusal("Tokenizer-less related models require a safetensors checkpoint.")
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
                original_vocab_size=original_vocab_size,
                model_vocab_size=checkpoint_vocab_size,
                source_dir=source_dir if source_dir is not None and source_dir.is_dir() else None,
            )
    except PruneRefusal as exc:
        raise SystemExit(str(exc)) from exc

    print("VocabPruner Prune")
    print(f"Model: {model}")
    print(f"Save: {result.output_dir}")
    print(f"Original vocab size: {original_vocab_size}")
    print(f"Retained vocab size: {result.retained_count}")
    print(f"Pruned vocab size: {result.pruned_count}")
    if result.original_checkpoint_bytes is not None and result.pruned_checkpoint_bytes is not None:
        reduction = result.original_checkpoint_bytes - result.pruned_checkpoint_bytes
        percentage = (reduction / result.original_checkpoint_bytes * 100.0) if result.original_checkpoint_bytes else 0.0
        print(
            "Checkpoint size: "
            f"{format_gib(result.original_checkpoint_bytes)} GiB -> "
            f"{format_gib(result.pruned_checkpoint_bytes)} GiB "
            f"({format_gib(reduction)} GiB, {percentage:.2f}% smaller)"
        )
    elif result.pruned_checkpoint_bytes is not None:
        print(f"Pruned checkpoint size: {format_gib(result.pruned_checkpoint_bytes)} GiB")
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
    write_checkpoint_manifest(save, model=model, snapshot_path=save)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"analyze", "prune"}:
        for name in args.selection_group:
            group = selection_groups()[name]
            args.keep_script.extend(group.keep_scripts)
            args.syntax_list.extend(group.syntax_lists)
            args.symbol_list.extend(group.symbol_lists)
    if args.command == "init":
        print(f"Environment file: {load_user_env(create=True)}")
    elif args.command == "inspect":
        inspect_command(args.model, trust_remote_code=args.trust_remote_code)
    elif args.command == "pull":
        pull_command(args.model, trust_remote_code=args.trust_remote_code)
    elif args.command == "analyze":
        analyze_command(
            model=args.model,
            keep_script=args.keep_script,
            syntax_list=args.syntax_list,
            symbol_list=args.symbol_list,
            dictionary=args.dictionary,
            languages=args.languages,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            seed=args.seed,
            workers=args.workers,
            show=args.show,
            create=args.create,
            recreate_list=args.recreate_list,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.command == "prune":
        prune_command(
            model=args.model,
            save=args.save,
            keep_script=args.keep_script,
            syntax_list=args.syntax_list,
            symbol_list=args.symbol_list,
            dictionary=args.dictionary,
            languages=args.languages,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            workers=args.workers,
            show=args.show,
            create=args.create,
            recreate_list=args.recreate_list,
            yes=args.yes,
            trust_remote_code=args.trust_remote_code,
            related_path=args.related_path,
            related_model=args.related_model,
            save_suffix=args.save_suffix,
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


def _is_model_downloaded(model: str) -> bool:
    return resolve_gguf(model) is not None or cached_checkpoint_status(model).complete


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
