from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vocabpruner.analysis.cache import (
    CACHE_SCHEMA_VERSION,
    list_settings,
    load_cached_facts,
    token_cache_dir,
    write_cached_facts,
)
from vocabpruner.analysis.dictionaries import load_dictionaries
from vocabpruner.analysis.scoring import (
    arrays_from_facts,
    derive_scores,
    ScoreConfig,
    token_fact_row,
)
from vocabpruner.models import TokenRecord, TokenScore
from vocabpruner.tokenization.preservation import complete_retained_token_ids


@dataclass(frozen=True)
class PipelineResult:
    model: str
    records: list[TokenRecord]
    scores: list[TokenScore]
    retained_ids: list[int]
    pruned_ids: list[int]
    cache_dir: Path
    cache_reused: bool
    token_facts: dict[str, Any]
    wordfreq_facts: dict[str, Any]
    arrays: "PipelineArrays | None"


def run_token_pipeline(
    *,
    model: str,
    config: ScoreConfig,
    cache_root: str | Path | None = None,
    records: list[TokenRecord],
    tokenizer: Any = None,
    workers: int = 1,
    derive_decisions: bool = True,
    progress: bool = False,
    recreate_list: bool = False,
) -> PipelineResult:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    cache_dir = token_cache_dir(model, config, cache_root=cache_root)
    settings = list_settings(model, config)
    cached = None if recreate_list else load_cached_facts(cache_dir, settings)
    cache_reused = cached is not None
    if cached is None:
        token_facts, wordfreq_facts = build_token_facts(records, config, workers=workers, progress=progress)
        write_cached_facts(cache_dir, settings, token_facts, wordfreq_facts)
    else:
        token_facts, wordfreq_facts = cached
    arrays = arrays_from_facts(token_facts, wordfreq_facts, config) if derive_decisions else None
    scores = derive_scores(records, token_facts, wordfreq_facts, config, arrays) if arrays is not None else []
    retained_ids = [score.token.token_id for score in scores if score.status in {"protected", "retain_candidate"}]
    if tokenizer is not None and scores:
        completed_ids = complete_retained_token_ids(tokenizer, retained_ids)
        added_ids = set(completed_ids) - set(retained_ids)
        for score in scores:
            if score.token.token_id in added_ids:
                score.status = "protected"
                score.protected = True
                score.explanations.append("required byte token or BPE merge dependency of a retained token")
        retained_ids = completed_ids
    pruned_ids = [score.token.token_id for score in scores if score.status == "non_retain_candidate"]
    return PipelineResult(
        model=model,
        records=records,
        scores=scores,
        retained_ids=retained_ids,
        pruned_ids=pruned_ids,
        cache_dir=cache_dir,
        cache_reused=cache_reused,
        token_facts=token_facts,
        wordfreq_facts=wordfreq_facts,
        arrays=arrays,
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


def build_token_facts(
    records: list[TokenRecord],
    config: ScoreConfig,
    workers: int = 1,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dictionaries = load_dictionaries(list(config.custom_dictionaries))
    token_columns: dict[str, list[Any]] = {
        "token_id": [],
        "raw_token": [],
        "decoded_text": [],
        "is_special": [],
        "is_byte_fallback": [],
        "script_counts": [],
        "total_meaningful": [],
        "low_confidence": [],
        "protected": [],
        "gate_blocked": [],
        "retain_candidate": [],
        "has_word_segments": [],
        "evidence_segments": [],
        "dictionary_accepted": [],
        "syntax_lists": [],
        "symbol_lists": [],
        "dictionary_matches": [],
        "explanations": [],
    }
    wordfreq_columns: dict[str, list[Any]] = {
        "token_id": [],
        "segment_index": [],
        "language": [],
        "term": [],
        "mode": [],
        "zipf": [],
        "rank": [],
        "rank_percent": [],
    }
    facts = _token_fact_rows(records, config, dictionaries, workers=workers, progress=progress)
    for record, fact, wordfreq_rows in facts:
        for key, value in fact.items():
            token_columns[key].append(value)
        for row in wordfreq_rows:
            for key, value in row.items():
                wordfreq_columns[key].append(value)
    return (
        {"schema_version": CACHE_SCHEMA_VERSION, "columns": token_columns},
        {"schema_version": CACHE_SCHEMA_VERSION, "columns": wordfreq_columns},
    )


def _token_fact_rows(
    records: list[TokenRecord],
    config: ScoreConfig,
    dictionaries: dict[str, set[str]],
    workers: int,
    progress: bool,
) -> list[tuple[TokenRecord, dict[str, Any], list[dict[str, Any]]]]:
    if workers == 1 or len(records) <= 1:
        return [
            token_fact_row(record, config, dictionaries)
            for record in _progress(records, progress=progress, total=len(records))
        ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = executor.map(
            _token_fact_row_from_args,
            ((record, config, dictionaries) for record in records),
            chunksize=max(1, len(records) // (workers * 8)),
        )
        return list(_progress(rows, progress=progress, total=len(records)))


def _token_fact_row_from_args(args) -> tuple[TokenRecord, dict[str, Any], list[dict[str, Any]]]:
    record, config, dictionaries = args
    return token_fact_row(record, config, dictionaries)


def _progress(iterable, progress: bool, total: int):
    if not progress:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="Classifying token facts", unit="tok")
