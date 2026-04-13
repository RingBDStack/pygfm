"""LLaGA data/cache roots (GFM-Toolbox convention)."""
from __future__ import annotations

import os


def get_repo_root() -> str:
    """Repo root (directory containing pyproject.toml if present)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def get_llaga_data_root() -> str:
    """
    Graph data root; expect dataset/ subtree like upstream.

    LLAGA_DATA_ROOT env, else <repo>/datasets/llaga.
    """
    env = os.environ.get("LLAGA_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.join(get_repo_root(), "datasets", "llaga")


def dataset_rel(*parts: str) -> str:
    """Absolute path dataset/<parts>, e.g. (cora, processed_data.pt)."""
    return os.path.join(get_llaga_data_root(), "dataset", *parts)


def default_llaga_cora_pt_path() -> str:
    """Convention: <LLAGA_DATA_ROOT>/Cora.pt (PyG tuple/dict/Data)."""
    return os.path.join(get_llaga_data_root(), "Cora.pt")


def resolve_cora_data_path(graph_pt_path: str | None = None) -> str:
    """
    Cora graph path resolution order:

    1. Explicit ``graph_pt_path`` (CLI/YAML ``--graph_pt_path``)
    2. Env ``LLAGA_CORA_PT`` or ``LLAGA_GRAPH_PT`` (path to a .pt file)
    3. If ``<data_root>/Cora.pt`` exists, use that PyG-style file
    4. Else ``dataset/cora/processed_data.pt`` (full upstream LLaGA layout)

    Raise FileNotFoundError if none found.
    """
    if graph_pt_path:
        p = os.path.abspath(graph_pt_path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"graph_pt_path not found: {p}")
        return p
    for key in ("LLAGA_CORA_PT", "LLAGA_GRAPH_PT"):
        v = os.environ.get(key)
        if v:
            p = os.path.abspath(v.strip())
            if os.path.isfile(p):
                return p
    cora_pt = default_llaga_cora_pt_path()
    legacy = dataset_rel("cora", "processed_data.pt")
    if os.path.isfile(cora_pt):
        return cora_pt
    if os.path.isfile(legacy):
        return legacy
    raise FileNotFoundError(
        f"No Cora graph file found. Tried: {cora_pt!r}, {legacy!r}. "
        "Set LLAGA_DATA_ROOT or place Cora.pt / dataset/cora/processed_data.pt."
    )


def get_llaga_ckpt_root() -> str:
    """Train output root: LLAGA_CKPT_ROOT or <repo>/ckpts/llaga/checkpoints."""
    env = os.environ.get("LLAGA_CKPT_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.join(get_repo_root(), "ckpts", "llaga", "checkpoints")


def get_llaga_hf_cache() -> str:
    """HF cache: LLAGA_HF_CACHE or <repo>/ckpts/llaga/hf_cache."""
    env = os.environ.get("LLAGA_HF_CACHE")
    if env:
        return os.path.abspath(env)
    return os.path.join(get_repo_root(), "ckpts", "llaga", "hf_cache")


def get_llaga_pretrained_root() -> str:
    """Local base models: <repo>/ckpts/llaga/pretrained_models."""
    env = os.environ.get("LLAGA_PRETRAINED_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.join(get_repo_root(), "ckpts", "llaga", "pretrained_models")
