from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_MODELS_DIR = PROJECT_ROOT / "hf_models"
LISTS_DIR = PROJECT_ROOT / "lists"
DECISIONS_DIR = PROJECT_ROOT / "decisions"
PRUNED_MODELS_DIR = PROJECT_ROOT / "pruned_models"
