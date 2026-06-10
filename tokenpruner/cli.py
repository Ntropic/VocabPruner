from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tokenpruner.architecture import inspect_model_architecture, load_model_config, render_model_architecture
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
    filter_scores,
    render_jsonl,
    render_markdown,
    render_plain_text,
    should_show_examples,
    should_show_json,
    should_show_rows,
)
from tokenpruner.sampling import score_sample
from tokenpruner.scoring import ScoreConfig
from tokenpruner.token_lists import (
    compact_token_maps,
    load_token_id_file,
    score_config_to_dict,
    split_token_ids,
    write_token_list_artifacts,
)
from tokenpruner.vocab import load_tokenizer, tokenizer_records, vocabulary_stats

DEFAULT_SAVE = "__tokenpruner_default_save__"


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenpruner",
        description="Tokenizer vocabulary analysis and sample-first pruning validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect tokenizer vocabulary metadata.")
    inspect_parser.add_argument("--model", required=True, help="Hugging Face model ID or local tokenizer path.")
    inspect_parser.add_argument("--trust-remote-code", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Classify a random sample of tokenizer tokens.")
    analyze_parser.add_argument("--model", required=True, help="Hugging Face model ID or local tokenizer path.")
    add_policy_arguments(analyze_parser)
    analyze_parser.add_argument("--sample-size", type=int, default=-1)
    analyze_parser.add_argument("--seed", type=int, default=13)
    add_worker_argument(analyze_parser)
    add_show_argument(analyze_parser)
    add_save_argument(analyze_parser, "Directory for analyze report artifacts.")
    analyze_parser.add_argument("--trust-remote-code", action="store_true")

    list_parser = subparsers.add_parser("list", help="Classify the full vocabulary and write token ID lists.")
    list_parser.add_argument("--model", required=True, help="Hugging Face model ID or local tokenizer path.")
    add_save_argument(list_parser, "Directory for token list artifacts.", default=DEFAULT_SAVE)
    add_policy_arguments(list_parser)
    add_worker_argument(list_parser)
    list_parser.add_argument("--trust-remote-code", action="store_true")

    prune_parser = subparsers.add_parser(
        "prune",
        help="Prune supported language or multimodal models with compact text-token IDs.",
    )
    prune_parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path.")
    add_save_argument(prune_parser, "Directory for the pruned model.", default=DEFAULT_SAVE)
    prune_parser.add_argument(
        "--retained-token-ids",
        help="Optional retained_token_ids.txt from tokenpruner list. If omitted, the policy is classified first.",
    )
    prune_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the pruning confirmation prompt.",
    )
    add_policy_arguments(prune_parser)
    add_worker_argument(prune_parser)
    add_show_argument(prune_parser)
    prune_parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keep-script", nargs="+", default=["Latin", "Common", "Inherited"])
    parser.add_argument("--symbol-preset", nargs="+", default=["coding", "ascii-punctuation"])
    parser.add_argument("--dictionary", action="append", default=[], help="Custom dictionary text file.")
    parser.add_argument("--wordfreq-lang", nargs="+", default=["en"])
    parser.add_argument("--wordfreq-min-zipf", type=float)
    parser.add_argument("--wordfreq-top-pct", type=float)


def add_worker_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Process workers for token classification. Defaults to CPU count minus 2.",
    )


def add_save_argument(parser: argparse.ArgumentParser, help_text: str, default: str | None = None) -> None:
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_SAVE,
        default=default,
        metavar="DIR",
        help=f"{help_text} Omit DIR to use the generated default path.",
    )


def add_show_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--show",
        nargs="+",
        action="append",
        metavar="SHOW",
        default=None,
        help=(
            "What to print: stats, all, examples, json. "
            "Can be combined. Examples accepts examples, examples:10, or "
            "examples:retained=20,pruned=20."
        ),
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
    wordfreq_lang: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    sample_size: int,
    seed: int,
    workers: int,
    show: list[str],
    save: str | None,
    trust_remote_code: bool = False,
) -> None:
    if sample_size == 0 or sample_size < -1:
        raise SystemExit("--sample-size must be -1 or at least 1")
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    show, example_counts = normalize_show_args(show)
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    config = policy_config(
        keep_script,
        symbol_preset,
        dictionary,
        wordfreq_lang,
        wordfreq_min_zipf,
        wordfreq_top_pct,
    )
    scores = score_sample(records, config=config, sample_size=sample_size, seed=seed, workers=workers, progress=True)
    visible_scores = filter_scores(scores, show)
    if save:
        save = resolve_save_dir(
            "analyze",
            save,
            model=model,
            config=config,
            sample_size=sample_size,
            seed=seed,
        )
        write_analyze_output(save, model=model, records=records, config=config, scores=scores, sample_size=sample_size, seed=seed)
    if should_show_rows(show) or should_show_examples(show) or "stats" in set(show):
        report = render_plain_text(
            model, records, scores, sample_size, seed, show=show, example_counts=example_counts, style="rich"
        )
        try:
            from rich.console import Console

            Console().print(report, soft_wrap=True)
        except ImportError:
            print(render_plain_text(model, records, scores, sample_size, seed, show=show, example_counts=example_counts))
    if should_show_json(show):
        jsonl = render_jsonl(visible_scores)
        if jsonl:
            print(jsonl)


def list_command(
    model: str,
    save: str,
    keep_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    wordfreq_lang: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    workers: int,
    trust_remote_code: bool = False,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    config = policy_config(
        keep_script,
        symbol_preset,
        dictionary,
        wordfreq_lang,
        wordfreq_min_zipf,
        wordfreq_top_pct,
    )
    save = resolve_save_dir("list", save, model=model, config=config)
    scores = score_sample(records, config=config, sample_size=-1, seed=13, workers=workers, progress=True)
    write_token_list_artifacts(save, model=model, config=config, scores=scores)
    retained_ids, pruned_ids = split_token_ids(scores)
    print("TokenPruner Token Lists")
    print(f"Model/tokenizer: {model}")
    print(f"Save: {save}")
    print(f"Classified tokens: {len(scores)}")
    print(f"Retained: {len(retained_ids)}")
    print(f"Pruned: {len(pruned_ids)}")


def prune_command(
    model: str,
    save: str,
    retained_token_ids: str | None,
    keep_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    wordfreq_lang: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
    workers: int,
    show: list[str] | list[list[str]] | None = None,
    yes: bool = False,
    trust_remote_code: bool = False,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    show, example_counts = normalize_show_args(show, default=["stats"])
    if retained_token_ids and set(show) != {"stats"}:
        raise SystemExit(
            "--show examples/rows/json requires full classification; omit --retained-token-ids."
        )
    model_config = load_model_config(model, trust_remote_code=trust_remote_code)
    architecture_report = inspect_model_architecture(model_config)
    try:
        validate_pruning_architecture(architecture_report, len(records))
    except PruneRefusal as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint_vocab_size = model_vocabulary_size(architecture_report, len(records))
    config = policy_config(
        keep_script,
        symbol_preset,
        dictionary,
        wordfreq_lang,
        wordfreq_min_zipf,
        wordfreq_top_pct,
    )
    save = resolve_save_dir(
        "prune",
        save,
        model=model,
        config=config,
        retained_token_ids=retained_token_ids,
    )
    scores = None
    if retained_token_ids:
        retained_ids = load_token_id_file(retained_token_ids)
    else:
        scores = score_sample(records, config=config, sample_size=-1, seed=13, workers=workers, progress=True)
        write_token_list_artifacts(Path(save) / "token_lists", model=model, config=config, scores=scores)
        retained_ids, _ = split_token_ids(scores)

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
        if scores is not None:
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
            wordfreq_lang=args.wordfreq_lang,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            sample_size=args.sample_size,
            seed=args.seed,
            workers=args.workers,
            show=args.show,
            save=args.save,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.command == "list":
        list_command(
            model=args.model,
            save=args.save,
            keep_script=args.keep_script,
            symbol_preset=args.symbol_preset,
            dictionary=args.dictionary,
            wordfreq_lang=args.wordfreq_lang,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            workers=args.workers,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.command == "prune":
        prune_command(
            model=args.model,
            save=args.save,
            retained_token_ids=args.retained_token_ids,
            keep_script=args.keep_script,
            symbol_preset=args.symbol_preset,
            dictionary=args.dictionary,
            wordfreq_lang=args.wordfreq_lang,
            wordfreq_min_zipf=args.wordfreq_min_zipf,
            wordfreq_top_pct=args.wordfreq_top_pct,
            workers=args.workers,
            show=args.show,
            yes=args.yes,
            trust_remote_code=args.trust_remote_code,
        )


def app() -> None:
    main()


def policy_config(
    keep_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    wordfreq_lang: list[str],
    wordfreq_min_zipf: float | None,
    wordfreq_top_pct: float | None,
) -> ScoreConfig:
    try:
        return ScoreConfig.from_cli(
            keep_scripts=keep_script,
            symbol_presets=symbol_preset,
            wordfreq_languages=wordfreq_lang,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
            dictionaries=dictionary,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def write_analyze_output(
    save: str,
    *,
    model: str,
    records,
    config: ScoreConfig,
    scores,
    sample_size: int,
    seed: int,
) -> dict[str, Path]:
    output_dir = Path(save)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "decisions": output_dir / "token_decisions.jsonl",
        "summary": output_dir / "summary.md",
        "policy": output_dir / "policy.json",
    }
    paths["decisions"].write_text(render_jsonl(scores) + ("\n" if scores else ""), encoding="utf-8")
    paths["summary"].write_text(render_markdown(model, records, scores, sample_size, seed, show=["all"]), encoding="utf-8")
    paths["policy"].write_text(
        json.dumps(
            {
                "model": model,
                "policy": score_config_to_dict(config),
                "counts": {"classified": len(scores)},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def resolve_save_dir(
    command: str,
    save: str,
    *,
    model: str,
    config: ScoreConfig,
    sample_size: int | None = None,
    seed: int | None = None,
    retained_token_ids: str | None = None,
) -> str:
    if save != DEFAULT_SAVE:
        return save

    parts = [
        model_slug(model),
        f"scripts-{value_slug(config.keep_scripts)}",
        f"symbols-{value_slug(config.symbol_presets)}",
        f"langs-{value_slug(config.wordfreq_languages)}",
        threshold_slug(config),
    ]
    if sample_size is not None:
        sample_label = "all" if sample_size == -1 else str(sample_size)
        parts.append(f"sample-{sample_label}")
    if seed is not None:
        parts.append(f"seed-{seed}")
    if retained_token_ids:
        parts.append(f"ids-{model_slug(Path(retained_token_ids).stem)}")
    name = "__".join(part for part in parts if part)
    return str(Path("tokenpruner_artifacts") / command / name)


def threshold_slug(config: ScoreConfig) -> str:
    if config.wordfreq_top_pct is not None:
        return f"top-pct-{number_slug(config.wordfreq_top_pct)}"
    return f"zipf-{number_slug(config.wordfreq_min_zipf)}"


def value_slug(values: list[str]) -> str:
    if not values:
        return "none"
    return "-".join(model_slug(value) for value in sorted(values))


def number_slug(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def model_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "unnamed"


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
            sample_size=-1,
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
                    sample_size=-1,
                    seed=13,
                    show=show,
                    example_counts=example_counts,
                )
            )
    if should_show_json(show):
        visible_scores = filter_scores(scores, show)
        jsonl = render_jsonl(visible_scores)
        if jsonl:
            print(jsonl)


def normalize_show_args(
    show: list[str] | list[list[str]] | None,
    default: list[str] | None = None,
) -> tuple[list[str], ExampleCounts]:
    if not show:
        return list(default or ["all"]), ExampleCounts()
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
    valid = {"all", "examples", "stats", "json"}
    unknown = [item for item in normalized if item not in valid]
    if unknown:
        raise SystemExit(f"Unsupported --show value(s): {', '.join(unknown)}")
    ordered = [
        item for item in ["stats", "all", "examples", "json"] if item in normalized
    ]
    return ordered or ["all"], example_counts


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
