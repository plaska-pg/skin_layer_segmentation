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
BASE_CONFIG_PATH = Path(os.environ.get("SKIN_SEG_CONFIG") or Path(__file__).resolve().parent / "base_config.yaml")


def load_config(path: Path = None) -> dict:
    """Parsed YAML config dict, or {} if the file doesn't exist."""
    p = Path(path) if path else BASE_CONFIG_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


CONFIG = load_config()
