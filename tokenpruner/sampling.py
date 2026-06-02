from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor

from tokenpruner.dictionaries import load_dictionaries
from tokenpruner.models import TokenRecord, TokenScore
from tokenpruner.scoring import ScoreConfig, score_token


def sample_records(records: list[TokenRecord], sample_size: int, seed: int) -> list[TokenRecord]:
    if sample_size == -1:
        return list(records)
    if sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def score_sample(
    records: list[TokenRecord],
    config: ScoreConfig,
    sample_size: int,
    seed: int,
    workers: int = 1,
    progress: bool = False,
) -> list[TokenScore]:
    sampled = sample_records(records, sample_size, seed)
    dictionaries = load_dictionaries(
        list(config.custom_dictionaries),
        list(config.builtin_dictionaries) if config.builtin_dictionaries is not None else None,
    )
    if workers <= 1 or len(sampled) <= 1:
        iterable = _progress(sampled, progress=progress, total=len(sampled))
        return [score_token(record, config, dictionaries=dictionaries) for record in iterable]

    tasks = [(record, config, dictionaries) for record in sampled]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        mapped = executor.map(_score_token_process, tasks)
        return list(_progress(mapped, progress=progress, total=len(tasks)))


def _score_token_process(task: tuple[TokenRecord, ScoreConfig, dict[str, set[str]]]) -> TokenScore:
    record, config, dictionaries = task
    return score_token(record, config, dictionaries=dictionaries)


def _progress(iterable, progress: bool, total: int):
    if not progress:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="Scoring tokens", unit="tok")
