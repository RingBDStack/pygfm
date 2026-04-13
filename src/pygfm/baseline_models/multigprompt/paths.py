from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    # pygfm/baseline_models/multigprompt/paths.py -> repo root
    return Path(__file__).resolve().parents[3]


def get_datasets_root() -> Path:
    return get_repo_root() / "datasets" / "multigprompt"


def get_ckpts_root() -> Path:
    return get_repo_root() / "ckpts" / "multigprompt" / "checkpoints"


def _legacy_combined_data_dir() -> Path | None:
    """
    If set, upstream (Planetoid) and downstream (few-shot) share one directory
    (legacy layout; same behavior as old ``MULTIGPROMPT_DATA_DIR`` + ``.../data``).
    """
    v = os.environ.get("MULTIGPROMPT_DATA_DIR")
    if v:
        return Path(v).expanduser().resolve()
    return None


def get_upstream_data_dir(dataset: str) -> Path:
    """
    Upstream graph data directory, **one subfolder per dataset**::

        datasets/multigprompt/<dataset>/ind.<dataset>.*   # raw Planetoid pkl
        datasets/multigprompt/<dataset>/data.pt         # single PyG graph (same role as Cora.pt)
        datasets/multigprompt/<dataset>/Cora.pt

    You may also place ``Cora.pt`` flat under ``datasets/multigprompt/`` (either layout is fine).

    ``dataset`` is typically ``cora`` / ``citeseer`` / ``pubmed`` (lowercased in paths).

    Override parent only: ``MULTIGPROMPT_UPSTREAM_DATA_DIR`` points to the parent of
    those folders (e.g. ``datasets/multigprompt``), and this function returns
    ``<parent>/<dataset>/``. Legacy ``MULTIGPROMPT_DATA_DIR`` uses a single flat dir.
    """
    ds = dataset.lower()
    if (p := _legacy_combined_data_dir()) is not None:
        return p
    v = os.environ.get("MULTIGPROMPT_UPSTREAM_DATA_DIR")
    if v:
        return Path(v).expanduser().resolve() / ds
    return get_datasets_root() / ds


def get_downstream_data_dir(dataset: str) -> Path:
    """
    Downstream few-shot + CSV for one dataset::

        downstream_data/multigprompt/<dataset>/fewshot_<dataset>/...
        downstream_data/multigprompt/<dataset>/<dataset>_fewshot.csv

    ``MULTIGPROMPT_DOWNSTREAM_DATA_DIR`` = parent of ``cora/``, ``citeseer/``, … folders.
    Legacy ``MULTIGPROMPT_DATA_DIR`` = single flat dir.
    """
    ds = dataset.lower()
    if (p := _legacy_combined_data_dir()) is not None:
        return p
    v = os.environ.get("MULTIGPROMPT_DOWNSTREAM_DATA_DIR")
    if v:
        return Path(v).expanduser().resolve() / ds
    return get_repo_root() / "downstream_data" / "multigprompt" / ds


def get_data_dir() -> Path:
    """Deprecated: use ``get_upstream_data_dir(\"cora\")`` or pass the dataset name."""
    return get_upstream_data_dir("cora")


def get_default_pretrain_ckpt_path(dataset: str) -> Path:
    return get_ckpts_root() / f"{dataset}.pkl"
