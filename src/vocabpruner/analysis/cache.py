from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from vocabpruner.analysis.scoring import ScoreConfig
from vocabpruner.analysis.scripts import compact_script_values, script_list_fingerprint
from vocabpruner.analysis.symbols import (
    compact_list_values,
    symbol_list_fingerprint,
    symbol_lists,
    syntax_list_fingerprint,
    syntax_lists,
)
from vocabpruner.storage.paths import LISTS_DIR


CACHE_SCHEMA_VERSION = 6


def token_cache_dir(model: str, config: ScoreConfig, cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else LISTS_DIR
    return root / compact_settings_slug(model, config, include_threshold=False)


def compact_settings_slug(model: str, config: ScoreConfig, include_threshold: bool = True) -> str:
    parts = [
        model_slug(model),
        f"scripts-{value_slug(sorted(config.keep_scripts))}",
        f"syntax-{list_slug(config.syntax_lists)}",
        f"symbols-{list_slug(config.symbol_lists)}",
        f"langs-{value_slug(config.languages)}",
    ]
    if config.custom_dictionaries:
        parts.append(f"dicts-{value_slug(config.custom_dictionaries)}")
    if include_threshold:
        parts.append(threshold_slug(config))
    return "__".join(parts)


def list_settings(model: str, config: ScoreConfig) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "settings": {
            "keep_scripts": compact_script_values(config.keep_scripts),
            "script_lists_fingerprint": script_list_fingerprint(config.keep_scripts),
            "syntax_lists": compact_list_values(config.syntax_lists, syntax_lists()),
            "syntax_lists_fingerprint": syntax_list_fingerprint(config.syntax_lists),
            "symbol_lists": compact_list_values(config.symbol_lists, symbol_lists()),
            "symbol_lists_fingerprint": symbol_list_fingerprint(config.symbol_lists),
            "languages": list(config.languages),
            "no_wordfreq": config.no_wordfreq,
            "custom_dictionaries": list(config.custom_dictionaries),
        },
    }


def load_cached_facts(cache_dir: Path, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    settings_path = cache_dir / "settings.json"
    token_path = cache_dir / "token_facts.json"
    wordfreq_path = cache_dir / "wordfreq_facts.json"
    if not settings_path.exists() or not token_path.exists() or not wordfreq_path.exists():
        return None
    try:
        cached_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached_settings != settings:
        return None
    return (
        json.loads(token_path.read_text(encoding="utf-8")),
        json.loads(wordfreq_path.read_text(encoding="utf-8")),
    )


def write_cached_facts(
    cache_dir: Path,
    settings: dict[str, Any],
    token_facts: dict[str, Any],
    wordfreq_facts: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (cache_dir / "token_facts.json").write_text(json.dumps(token_facts, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (cache_dir / "wordfreq_facts.json").write_text(json.dumps(wordfreq_facts, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def list_slug(values: set[str]) -> str:
    return value_slug(sorted(values))


def threshold_slug(config: ScoreConfig) -> str:
    if config.no_wordfreq:
        return "no-wordfreq"
    if config.wordfreq_top_pct is not None:
        return f"top-pct-{number_slug(config.wordfreq_top_pct)}"
    return f"zipf-{number_slug(config.wordfreq_min_zipf)}"


def value_slug(values) -> str:
    value_list = list(values)
    if not value_list:
        return "none"
    return "-".join(model_slug(str(value)) for value in sorted(value_list))


def number_slug(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def model_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "unnamed"
