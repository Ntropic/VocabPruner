from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenpruner.models import TokenScore
from tokenpruner.reports import render_jsonl, score_stats
from tokenpruner.scoring import ScoreConfig

RETAINED_STATUSES = {"protected", "retain_candidate"}


def is_retained_decision(score: TokenScore) -> bool:
    return score.status in RETAINED_STATUSES


def split_token_ids(scores: list[TokenScore]) -> tuple[list[int], list[int]]:
    retained = sorted(score.token.token_id for score in scores if is_retained_decision(score))
    pruned = sorted(score.token.token_id for score in scores if not is_retained_decision(score))
    return retained, pruned


def compact_token_maps(retained_token_ids: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    ordered = sorted(int(token_id) for token_id in retained_token_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("retained token IDs must be unique")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(ordered)}
    new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
    return old_to_new, new_to_old


def write_token_list_artifacts(
    output_dir: str | Path,
    *,
    model: str,
    config: ScoreConfig,
    scores: list[TokenScore],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    retained_ids, pruned_ids = split_token_ids(scores)

    paths = {
        "retained": output / "retained_token_ids.txt",
        "pruned": output / "pruned_token_ids.txt",
        "decisions": output / "token_decisions.jsonl",
        "policy": output / "policy.json",
    }
    paths["retained"].write_text(_token_id_lines(retained_ids), encoding="utf-8")
    paths["pruned"].write_text(_token_id_lines(pruned_ids), encoding="utf-8")
    paths["decisions"].write_text(render_jsonl(scores) + ("\n" if scores else ""), encoding="utf-8")
    paths["policy"].write_text(
        json.dumps(
            {
                "model": model,
                "policy": score_config_to_dict(config),
                "counts": {
                    **score_stats(scores),
                    "classified": len(scores),
                    "retained_ids": len(retained_ids),
                    "pruned_ids": len(pruned_ids),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def load_token_id_file(path: str | Path) -> list[int]:
    token_ids: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token_ids.append(int(stripped))
    return token_ids


def score_config_to_dict(config: ScoreConfig) -> dict[str, Any]:
    return {
        "keep_scripts": sorted(config.keep_scripts),
        "drop_scripts": sorted(config.drop_scripts),
        "symbol_presets": sorted(config.symbol_presets),
        "wordfreq_languages": list(config.wordfreq_languages),
        "wordfreq_min_zipf": config.wordfreq_min_zipf,
        "wordfreq_top_pct": config.wordfreq_top_pct,
        "custom_dictionaries": list(config.custom_dictionaries),
        "builtin_dictionaries": (
            list(config.builtin_dictionaries)
            if config.builtin_dictionaries is not None
            else None
        ),
    }


def _token_id_lines(token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return "\n".join(str(token_id) for token_id in token_ids) + "\n"
