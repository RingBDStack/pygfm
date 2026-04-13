"""
OneForAll paths inside GFM-Toolbox.

Override with environment variables for production:
  ONEFORALL_DATA_ROOT   — user PyG graph assets root (default: <repo>/datasets/oneforall)
  ONEFORALL_CACHE_ROOT  — preprocessed/encoded OFA cache (default: <data_root>/cache_data)
  ONEFORALL_EXP_ROOT    — experiment logs (default: <repo>/ckpts/oneforall/runs)
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    """GFM-Toolbox repo root (directory containing `pygfm/` and `datasets/`)."""
    p = _PKG_DIR
    for _ in range(8):
        if (p / "pyproject.toml").exists():
            return p
        if (p / "pygfm").is_dir() and (p / "datasets").is_dir():
            return p
        p = p.parent
    # paths.py lives at <repo>/pygfm/baseline_models/oneforall/
    return _PKG_DIR.parents[2]


def get_data_root() -> Path:
    """User graph data root (e.g. ``Cora.pt``); default ``<repo>/datasets/oneforall``.

    Preprocessed OFA cache: :func:`get_cache_root` (default ``<data_root>/cache_data``).
    """
    env = os.environ.get("ONEFORALL_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "datasets" / "oneforall"


def get_cache_root() -> Path:
    env = os.environ.get("ONEFORALL_CACHE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return get_data_root() / "cache_data"


def get_model_cache_dir() -> Path:
    return get_cache_root() / "model"


def get_molecule_dataset_cache_dir() -> Path:
    return get_cache_root() / "dataset"


def get_exp_root() -> Path:
    env = os.environ.get("ONEFORALL_EXP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "ckpts" / "oneforall" / "runs"


def low_resource_split_json() -> Path:
    """Few-shot / low-resource split config (decoupled from code; under downstream_data)."""
    return _repo_root() / "downstream_data" / "oneforall" / "low_resource_split.json"


def ensure_runtime_dirs() -> None:
    get_data_root().mkdir(parents=True, exist_ok=True)
    get_cache_root().mkdir(parents=True, exist_ok=True)
    get_model_cache_dir().mkdir(parents=True, exist_ok=True)
    get_exp_root().mkdir(parents=True, exist_ok=True)
