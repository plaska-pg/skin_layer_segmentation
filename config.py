"""Loads base_config.yaml (repo root) - the single source of truth for
hyperparameters previously hardcoded across predict.py/postprocess.py/
measurement.py. Each of those modules reads its own top-level section
(predict/postprocess/measurement) at import time, falling back to its
previous hardcoded value if the key or file is missing.
"""

import os
from pathlib import Path

import yaml

# SKIN_SEG_CONFIG lets a caller (e.g. interface/app.py) point a run at an edited copy of the config
REPO_ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = Path(os.environ.get("SKIN_SEG_CONFIG") or REPO_ROOT / "base_config.yaml")


def repo_path(p) -> str:
    """p as given if absolute or present relative to the cwd, else relative to the repo root."""
    p = Path(p)
    return str(p if p.is_absolute() or p.exists() else REPO_ROOT / p)


def load_config(path: Path = None) -> dict:
    """Parsed YAML config dict, or {} if the file doesn't exist."""
    p = Path(path) if path else BASE_CONFIG_PATH
    if not p.exists():
        if not path and os.environ.get("SKIN_SEG_CONFIG"):
            raise FileNotFoundError(f"SKIN_SEG_CONFIG points to a missing file: {p}")
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


CONFIG = load_config()
