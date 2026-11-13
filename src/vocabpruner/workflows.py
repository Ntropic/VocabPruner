from __future__ import annotations

from pathlib import Path

from vocabpruner.analysis.pipeline import cached_create_only_pipeline, run_token_pipeline
from vocabpruner.analysis.scoring import settings_config
from vocabpruner.checkpoints.architecture import (
    inspect_model_architecture,
    load_model_config,
    model_vocabulary_size,
    render_model_architecture,
    validate_pruning_architecture,
)
from vocabpruner.checkpoints.codebooks import codebook_config_ids
from vocabpruner.checkpoints.gguf import prune_gguf_checkpoint
from vocabpruner.checkpoints.runtime import ensure_pruning_runtime
from vocabpruner.checkpoints.safetensors import prune_safetensors_checkpoint
from vocabpruner.checkpoints.torch import load_causal_lm, prune_model_and_tokenizer
from vocabpruner.errors import PruneRefusal, VocabRewriteError
from vocabpruner.mapping import compact_token_maps
from vocabpruner.presentation.artifacts import (
    resolve_artifact_dir,
    start_html_server,
    write_create_artifacts,
)
from vocabpruner.presentation.console import (
    confirm_pruning,
    normalize_create_args,
    normalize_show_args,
    print_cache_status,
    print_prune_decisions,
    print_pruning_preview,
)
from vocabpruner.presentation.reports import (
    format_gib,
    render_plain_text,
    should_show_examples,
    should_show_rows,
)
from vocabpruner.storage.gguf import resolve_gguf
from vocabpruner.storage.hub import (
    cached_checkpoint_status,
    resolve_checkpoint_source,
    write_checkpoint_manifest,
)
from vocabpruner.storage.metadata import (
    load_related_token_ids,
    resolve_related_model_dir,
    save_related_tokenizer,
    write_prune_metadata,
)
from vocabpruner.tokenization.loading import load_tokenizer, tokenizer_records, vocabulary_stats


DEFAULT_SAVE = "__vocabpruner_default_save__"


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
            tokenizer=tokenizer,
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
            scores=scores,
            cache_dir=result.cache_dir,
            html_dir=result.cache_dir,
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
                tokenizer=tokenizer,
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
                scores=scores,
                cache_dir=result.cache_dir,
                html_dir=result.cache_dir,
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
    except (PruneRefusal, VocabRewriteError) as exc:
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
    if related_dir is not None:
        save_related_tokenizer(related_dir, Path(save), result.retained_count)
    metadata_source = cached_checkpoint_status(model).snapshot_path
    write_prune_metadata(
        save,
        source_model=str(metadata_source if metadata_source is not None and metadata_source.exists() else model),
        score_config=config,
        related_model=related_dir,
    )
    write_checkpoint_manifest(save, model=model, snapshot_path=save)


def _is_model_downloaded(model: str) -> bool:
    return resolve_gguf(model) is not None or cached_checkpoint_status(model).complete
