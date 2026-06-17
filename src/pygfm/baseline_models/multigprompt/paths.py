from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """
    User project root (``~/gfm``), not the ``pygfm`` install directory.

    ``parents[3]`` only works for a full git checkout with ``repo/pygfm/...`` or ``repo/src/pygfm``;
    under ``pip install``, ``__file__`` lives in ``site-packages`` and that heuristic points at
    ``site-packages`` — then ``datasets/multigprompt`` resolves incorrectly.
    """
    for key in ("PYGFM_REPO_ROOT", "MULTIGPROMPT_REPO_ROOT"):
        v = os.environ.get(key)
        if v:
            p = Path(v).expanduser().resolve()
            if p.is_dir():
                return p
    here = Path(__file__).resolve()
    if "site-packages" in here.parts:
        return Path.cwd().resolve()
    for anc in here.parents:
        if not (anc / "pyproject.toml").is_file():
            continue
        if (anc / "src" / "pygfm").is_dir() or (anc / "pygfm" / "baseline_models").is_dir():
            return anc
    return Path.cwd().resolve()


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

    With ``pip install`` (no checkout), set ``PYGFM_REPO_ROOT`` / ``MULTIGPROMPT_REPO_ROOT`` to
    your project root (e.g. ``~/gfm``) so ``datasets/multigprompt`` resolves correctly; otherwise
    :func:`get_repo_root` falls back to :func:`Path.cwd`.
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
