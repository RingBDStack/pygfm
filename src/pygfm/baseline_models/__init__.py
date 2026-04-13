from __future__ import annotations

from importlib import import_module
from typing import Any

from pygfm.public.model_bases import (
    GFMDownPromptGraphModelBase,
    GFMDownPromptNodeModelBase,
    GFMLegacyModelBase,
    GFMModelBase,
    GFMModelDescriptor,
    GFMPrePromptModelBase,
    GFMStage,
)


def _lazy_import(module: str) -> Any:
    return import_module(f"{__name__}.{module}")


def __getattr__(name: str) -> Any:
    """
    Avoid importing every baseline (and their optional assets) at module import time.
    This keeps `import pygfm` and `import pygfm.baseline_models` robust.
    """
    # submodules
    if name in {
        "mdgpt",
        "rag_gfm",
        "graphgpt",
        "graphtext",
        "sa2gfm",
        "oneforall",
        "llaga",
        "multigprompt",
        "bridge",
        "samgpt",
        "mdgfm",
        "gcot",
        "graphkeeper",
        "graphprompt",
        "hgprompt",
        "bim_gfm",
        "classic_gnn",
        "graver",
        "graphmore",
    }:
        return _lazy_import(name)

    # commonly re-exported symbols (imported lazily)
    if name in {"PrePromptModel", "DownPromptModel", "DownPromptGraphModel", "MultiDomainGFM"}:
        mod = _lazy_import("mdgpt")
        return getattr(mod, name)

    if name in {"GraphLlamaForCausalLM", "load_model_pretrained", "transfer_param_tograph"}:
        mod = _lazy_import("graphgpt")
        return getattr(mod, name)

    if name in {
        "GraphKeeperPrePromptModel",
        "GraphKeeperDownPromptModel",
        "GraphKeeperDownPromptGraphModel",
        "set_incremental_optimizer",
    }:
        mod = _lazy_import("graphkeeper")
        return getattr(mod, name)

    if name in {"BridgePrePromptModel", "BridgeDownPromptModel", "BridgeDownPromptGraphModel"}:
        mod = _lazy_import("bridge")
        return getattr(mod, name)

    if name in {"SAMGPTPrePromptModel", "SAMGPTDownPromptModel", "SAMGPTDownPromptGraphModel"}:
        mod = _lazy_import("samgpt")
        return getattr(mod, name)

    if name in {"MDGFMPrePromptModel", "MDGFMDownPromptModel", "MDGFMDownPromptGraphModel"}:
        mod = _lazy_import("mdgfm")
        return getattr(mod, name)

    if name in {"GCoTPrePromptModel", "GCoTDownPromptModel", "GCoTDownPromptGraphModel"}:
        mod = _lazy_import("gcot")
        return getattr(mod, name)

    if name in {"GraphPromptPrePromptModel", "GraphPromptDownPromptModel", "GraphPromptDownPromptGraphModel"}:
        mod = _lazy_import("graphprompt")
        return getattr(mod, name)

    if name in {"HGPromptPrePromptModel", "HGPromptDownPromptModel"}:
        mod = _lazy_import("hgprompt")
        return getattr(mod, name)

    if name in {"BimGFMPrePromptModel", "BimGFMDownPromptModel"}:
        mod = _lazy_import("bim_gfm")
        return getattr(mod, name)

    if name in {"ClassicGNNPrePromptModel", "ClassicGNNDownPromptModel", "build_sparse_encoder"}:
        mod = _lazy_import("classic_gnn")
        return getattr(mod, name)

    if name in {"GRAVERPrePromptModel", "GRAVERDownPromptModel", "GRAVERDownPromptGraphModel"}:
        mod = _lazy_import("graver")
        return getattr(mod, name)

    if name in {"GraphMoREPrePromptModel", "GraphMoREDownPromptModel", "GraphMoREDownPromptGraphModel"}:
        mod = _lazy_import("graphmore")
        return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GFMStage",
    "GFMModelDescriptor",
    "GFMModelBase",
    "GFMPrePromptModelBase",
    "GFMDownPromptNodeModelBase",
    "GFMDownPromptGraphModelBase",
    "GFMLegacyModelBase",
    # Keep the rest lazily available via __getattr__ to avoid import-time failures.
]
