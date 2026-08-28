from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_path

VOCABPRUNER_CACHE_DIR = user_cache_path("vocabpruner")
LISTS_DIR = VOCABPRUNER_CACHE_DIR / "lists"
DECISIONS_DIR = VOCABPRUNER_CACHE_DIR / "decisions"


def hf_models_dir() -> Path:
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache).expanduser()
    if home := os.environ.get("HF_HOME"):
        return Path(home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def pruned_models_dir() -> Path:
    return Path(os.environ.get("VOCABPRUNER_SAVE_DIR", Path.cwd())).expanduser().resolve()
