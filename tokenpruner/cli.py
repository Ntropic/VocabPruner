from __future__ import annotations

import argparse
import os
from pathlib import Path

from tokenpruner.reports import (
    filter_scores,
    render_jsonl,
    render_plain_text,
    should_show_ambiguous_words,
    should_show_json,
    should_show_rows,
)
from tokenpruner.sampling import score_sample
from tokenpruner.scoring import ScoreConfig
from tokenpruner.vocab import load_tokenizer, tokenizer_records, vocabulary_stats


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

    analyze_parser = subparsers.add_parser("analyze", help="Score a random sample of tokenizer tokens.")
    analyze_parser.add_argument("--model", required=True, help="Hugging Face model ID or local tokenizer path.")
    analyze_parser.add_argument("--keep-script", nargs="+", default=["Latin", "Common", "Inherited"])
    analyze_parser.add_argument("--drop-script", nargs="+", default=[])
    analyze_parser.add_argument("--symbol-preset", nargs="+", default=["coding", "ascii-punctuation"])
    analyze_parser.add_argument("--dictionary", action="append", default=[], help="Custom dictionary text file.")
    analyze_parser.add_argument("--wordfreq-lang", nargs="+", default=["en"])
    analyze_parser.add_argument("--wordfreq-min-zipf", type=float, default=4.0)
    analyze_parser.add_argument("--wordfreq-top-pct", type=float)
    analyze_parser.add_argument("--sample-size", type=int, default=-1)
    analyze_parser.add_argument("--seed", type=int, default=13)
    analyze_parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Process workers for sampled token scoring. Defaults to CPU count minus 2.",
    )
    analyze_parser.add_argument(
        "--show",
        nargs="+",
        action="append",
        metavar="SHOW",
        default=None,
        help="What to print/save: all, ambiguous, ambiguous-words, stats, json. Can be combined.",
    )
    analyze_parser.add_argument("--save", help="Save selected token rows as JSONL to this file.")
    analyze_parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def inspect_command(model: str, trust_remote_code: bool = False) -> None:
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    stats = vocabulary_stats(records)

    print("Tokenizer Vocabulary")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


def analyze_command(
    model: str,
    keep_script: list[str],
    drop_script: list[str],
    symbol_preset: list[str],
    dictionary: list[str],
    wordfreq_lang: list[str],
    wordfreq_min_zipf: float,
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
    show = normalize_show_args(show)
    tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
    records = tokenizer_records(tokenizer)
    config = ScoreConfig.from_cli(
        keep_scripts=keep_script,
        drop_scripts=drop_script,
        symbol_presets=symbol_preset,
        wordfreq_languages=wordfreq_lang,
        wordfreq_min_zipf=wordfreq_min_zipf,
        wordfreq_top_pct=wordfreq_top_pct,
        dictionaries=dictionary,
    )
    scores = score_sample(records, config=config, sample_size=sample_size, seed=seed, workers=workers, progress=True)
    visible_scores = filter_scores(scores, show)
    saved_scores = save_scores(scores, show)
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(render_jsonl(saved_scores) + ("\n" if saved_scores else ""), encoding="utf-8")
    if should_show_rows(show) or should_show_ambiguous_words(show) or "stats" in set(show):
        report = render_plain_text(model, records, scores, sample_size, seed, show=show, style="rich")
        try:
            from rich.console import Console

            Console().print(report, soft_wrap=True)
        except ImportError:
            print(render_plain_text(model, records, scores, sample_size, seed, show=show))
    if should_show_json(show):
        jsonl = render_jsonl(visible_scores)
        if jsonl:
            print(jsonl)


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
            drop_script=args.drop_script,
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


def app() -> None:
    main()


def save_scores(scores, show: list[str]):
    show_set = set(show)
    if "ambiguous" in show_set and "all" not in show_set:
        return filter_scores(scores, ["ambiguous"])
    return list(scores)


def normalize_show_args(show: list[str] | list[list[str]] | None) -> list[str]:
    if not show:
        return ["all"]
    flattened: list[str] = []
    for item in show:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    valid = {"all", "ambiguous", "ambiguous-words", "stats", "json"}
    unknown = [item for item in flattened if item not in valid]
    if unknown:
        raise SystemExit(f"Unsupported --show value(s): {', '.join(unknown)}")
    ordered = [item for item in ["stats", "all", "ambiguous", "ambiguous-words", "json"] if item in flattened]
    return ordered or ["all"]


if __name__ == "__main__":
    main()
