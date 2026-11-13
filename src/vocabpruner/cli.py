from __future__ import annotations

import argparse
import os

from vocabpruner.analysis.groups import selection_groups
from vocabpruner.analysis.scripts import script_lists
from vocabpruner.analysis.symbols import symbol_lists, syntax_lists
from vocabpruner.storage.paths import load_user_env
from vocabpruner.workflows import (
    analyze_command,
    DEFAULT_SAVE,
    inspect_command,
    prune_command,
    pull_command,
)


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


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


if __name__ == "__main__":
    main()
