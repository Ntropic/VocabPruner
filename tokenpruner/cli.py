from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from tokenpruner.architecture import inspect_model_architecture, load_model_config, render_model_architecture
from tokenpruner.paths import DECISIONS_DIR, LISTS_DIR, PROJECT_ROOT, PRUNED_MODELS_DIR
from tokenpruner.pipeline import (
    PipelineResult,
    compact_settings_slug,
    list_settings,
    compact_symbol_values,
    load_cached_facts,
    run_token_pipeline,
    token_cache_dir,
    write_wordfreq_html,
)
from tokenpruner.pruning import (
    PruneRefusal,
    load_causal_lm,
    missing_pruning_packages,
    model_vocabulary_size,
    prune_model_and_tokenizer,
    prune_safetensors_checkpoint,
    validate_pruning_architecture,
    validate_pruning_runtime,
)
from tokenpruner.reports import (
    ExampleCounts,
    render_jsonl,
    render_plain_text,
    should_show_examples,
    should_show_rows,
)
from tokenpruner.scoring import ScoreConfig
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
    derive_decisions = bool(raw_show) or should_create_decisions(create) or not create
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
    decisions_dir = decision_artifact_dir(model=model, config=config, seed=seed)
    if create:
        write_create_artifacts(
            create,
            decisions_dir,
            scores=scores,
            cache_dir=result.cache_dir,
            config=config,
            html_dir=result.cache_dir,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
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
    save = resolve_save_dir("prune", save, model=model, config=config)
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
        save,
        scores=scores,
        cache_dir=result.cache_dir,
        config=config,
        html_dir=result.cache_dir,
        wordfreq_min_zipf=wordfreq_min_zipf,
        wordfreq_top_pct=wordfreq_top_pct,
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
            result = prune_model_and_tokenizer(
                model=loaded_model,
                tokenizer=tokenizer,
                output_dir=save,
                retained_token_ids=retained_ids,
                original_vocab_size=len(records),
                model_vocab_size=checkpoint_vocab_size,
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


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        inspect_command(args.model, trust_remote_code=args.trust_remote_code)
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


def settings_config(
    keep_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    languages: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
) -> ScoreConfig:
    try:
        return ScoreConfig.from_cli(
            keep_scripts=keep_script,
            symbol_presets=symbol_preset,
            languages=languages,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
            dictionaries=dictionary,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def require_wordfreq_threshold(wordfreq_min_zipf: float | None, wordfreq_top_pct: float | None) -> None:
    if wordfreq_min_zipf is None and wordfreq_top_pct is None:
        raise SystemExit(
            "A word frequency threshold is required for thresholded decisions. "
            "Use --wordfreq-min-zipf or --wordfreq-top-pct, or use only --create list."
        )


def cached_create_only_pipeline(model: str, config: ScoreConfig) -> PipelineResult | None:
    cache_dir = token_cache_dir(model, config)
    cached = load_cached_facts(cache_dir, list_settings(model, config))
    if cached is None:
        return None
    token_facts, wordfreq_facts = cached
    return PipelineResult(
        model=model,
        records=[],
        scores=[],
        retained_ids=[],
        pruned_ids=[],
        cache_dir=cache_dir,
        cache_reused=True,
        token_facts=token_facts,
        wordfreq_facts=wordfreq_facts,
        arrays=None,
    )


def write_token_decisions(save: str | Path, scores) -> Path:
    output_dir = Path(save)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "token_decisions.jsonl"
    path.write_text(render_jsonl(scores) + ("\n" if scores else ""), encoding="utf-8")
    return path


def write_create_artifacts(
    create: list[str],
    decisions_dir: str | Path,
    *,
    scores,
    cache_dir: Path,
    config: ScoreConfig,
    html_dir: str | Path,
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
) -> None:
    if should_create_list(create):
        print(f"List: {cache_dir}", file=sys.stderr)
    if should_create_decisions(create):
        decisions_path = write_token_decisions(decisions_dir, scores)
        print(f"Decisions: {decisions_path}", file=sys.stderr)
    if should_create_html(create):
        html_path = write_wordfreq_html(
            cache_dir,
            config,
            output_dir=html_dir,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
        )
        print(f"HTML: {html_path}")
        url, pid = start_html_server(Path(html_dir))
        print(f"Open: {url}")
        print(f"Server PID: {pid}", file=sys.stderr)


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


def score_config_to_dict(config: ScoreConfig) -> dict:
    return {
        "keep_scripts": sorted(config.keep_scripts),
        "symbol_presets": compact_symbol_values(config.symbol_presets),
        "languages": list(config.languages),
        "wordfreq_min_zipf": config.wordfreq_min_zipf,
        "wordfreq_top_pct": config.wordfreq_top_pct,
        "custom_dictionaries": list(config.custom_dictionaries),
        "builtin_dictionaries": (
            list(config.builtin_dictionaries)
            if config.builtin_dictionaries is not None
            else None
        ),
    }


def resolve_save_dir(
    command: str,
    save: str,
    *,
    model: str,
    config: ScoreConfig,
    seed: int | None = None,
) -> str:
    if save != DEFAULT_SAVE:
        path = Path(save)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    parts = [compact_settings_slug(model, config, include_threshold=(command != "list"))]
    if seed is not None:
        parts.append(f"seed-{seed}")
    name = "__".join(part for part in parts if part)
    root = PRUNED_MODELS_DIR if command == "prune" else LISTS_DIR / command
    return str(root / name)


def decision_artifact_dir(*, model: str, config: ScoreConfig, seed: int | None = None) -> str:
    parts = [compact_settings_slug(model, config, include_threshold=True)]
    if seed is not None:
        parts.append(f"seed-{seed}")
    return str(DECISIONS_DIR / "__".join(parts))


def format_gb(byte_count: int) -> str:
    return f"{byte_count / (1024 ** 3):.3f}"


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
        answer = input("Continue with model download and pruning? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def ensure_pruning_runtime(report, assume_yes: bool = False) -> None:
    missing = missing_pruning_packages(report)
    if not missing:
        return

    command = [sys.executable, "-m", "pip", "install", *missing]
    print("Missing packages required to prune this checkpoint:")
    for package in missing:
        print(f"- {package}")
    print("Suggested command:")
    print(" ".join(command))

    if not assume_yes:
        try:
            answer = input("Install these packages now? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            raise SystemExit("Required pruning packages were not installed.")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Package installation failed with exit code {exc.returncode}.") from exc

    try:
        validate_pruning_runtime(report)
    except PruneRefusal as exc:
        raise SystemExit(f"Packages were installed, but runtime validation still failed: {exc}") from exc


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


def should_create_list(create: list[str]) -> bool:
    return "list" in create


def should_create_decisions(create: list[str]) -> bool:
    return "decisions" in create


def should_create_html(create: list[str]) -> bool:
    return "html" in create


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
