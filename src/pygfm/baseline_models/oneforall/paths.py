"""
OneForAll paths inside GFM-Toolbox.

Override with environment variables for production:
  ONEFORALL_DATA_ROOT   — user PyG graph assets root (default: <repo>/datasets/oneforall)
  ONEFORALL_CACHE_ROOT  — preprocessed/encoded OFA cache (default: <data_root>/cache_data)
  ONEFORALL_EXP_ROOT    — experiment logs (default: <repo>/ckpts/oneforall/runs)
  PYGFM_REPO_ROOT       — checkout root when resolving paths from a pip install (optional)

Run YAML ``data_root`` is applied via :func:`configure_runtime_data_root` before ``setup_exp`` so
``./datasets/oneforall`` resolves under the process cwd (not ``site-packages``, which can contain
Hugging Face's unrelated ``datasets`` package).
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

# Set from run YAML ``data_root`` (run_cdm) so pip installs resolve Cora.pt under cwd, not site-packages.
_runtime_data_root: Path | None = None


def configure_runtime_data_root(data_root: str | Path | None) -> None:
    """Anchor graph assets (``Cora.pt``, …) from merged run config. Relative paths use :func:`os.getcwd`."""
    global _runtime_data_root
    if data_root is None:
        _runtime_data_root = None
        return
    s = str(data_root).strip()
    if s in ("", "null", "None"):
        _runtime_data_root = None
        return
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    _runtime_data_root = p


def _is_huggingface_datasets_package(datasets_dir: Path) -> bool:
    """``site-packages/datasets`` is the Hugging Face *library*, not GFM's data folder."""
    return (datasets_dir / "__init__.py").is_file()


def _repo_root() -> Path:
    """Project / checkout root for ``datasets/``, ``ckpts/`` (not ``site-packages``)."""
    p = _PKG_DIR
    for _ in range(14):
        if (p / "pyproject.toml").exists():
            return p
        if (p / "pygfm").is_dir() and (p / "datasets").is_dir():
            if not _is_huggingface_datasets_package(p / "datasets"):
                return p
        if (p / "src" / "pygfm").is_dir() and (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    env = os.environ.get("PYGFM_REPO_ROOT") or os.environ.get("ONEFORALL_PROJECT_ROOT")
    if env:
        ep = Path(env).expanduser().resolve()
        if ep.is_dir():
            return ep
    return Path.cwd().resolve()


def get_data_root() -> Path:
    """User graph data root (e.g. ``Cora.pt``); default ``<repo>/datasets/oneforall``.

    Preprocessed OFA cache: :func:`get_cache_root` (default ``<data_root>/cache_data``).
    """
    if _runtime_data_root is not None:
        return _runtime_data_root
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
