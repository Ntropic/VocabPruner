from __future__ import annotations

from pathlib import Path
import os


USER_ENV_PATH = Path.home() / ".vocabpruner" / ".env"


USER_ENV_KEYS = {"VOCABPRUNER_SAVE_DIR", "HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"}


USER_ENV_TEMPLATE = """\
# VocabPruner defaults. Remove # and edit paths to enable a setting.
# VOCABPRUNER_SAVE_DIR=/models/models
# HF_HUB_CACHE=/models/cache/huggingface/hub
# HF_HOME=/models/cache/huggingface
# XDG_CACHE_HOME=/models/cache
"""


def load_user_env(path: Path = USER_ENV_PATH, *, create: bool = False) -> Path:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(USER_ENV_TEMPLATE)
        except FileExistsError:
            pass
    if not path.is_file():
        return path
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if key in USER_ENV_KEYS:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key] = value
    return path


load_user_env()


VOCABPRUNER_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
).expanduser() / "vocabpruner"


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
