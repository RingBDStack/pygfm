"""Resolve ``(baseline, stage)`` → runner for ``pygfm -c config.yaml``."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

from pygfm.cli.baselines.stub_config import ALL_SCRIPT_PAIRS
from pygfm.cli.script_runner import make_runner

Runner = Callable[[dict[str, Any]], None]

_STAGE_ALIASES: dict[str, str] = {
    "ft": "finetune",
    "finetune_node": "finetune",
    "train": "pretrain",
}


def merge_config(cfg: dict[str, Any]) -> dict[str, Any]:
    params = cfg.get("params")
    base = {k: v for k, v in cfg.items() if k != "params"}
    if isinstance(params, dict):
        return {**params, **base}
    return base


def _discover_module_runners() -> dict[tuple[str, str], Runner]:
    out: dict[tuple[str, str], Runner] = {}
    import pygfm.cli.baselines as baselines_pkg

    for info in pkgutil.iter_modules(baselines_pkg.__path__):
        name = info.name
        if name.startswith("_") or name == "stub_config":
            continue
        mod = importlib.import_module(f"pygfm.cli.baselines.{name}")
        runners = getattr(mod, "RUNNERS", None)
        if not isinstance(runners, dict):
            continue
        for stage, fn in runners.items():
            if callable(fn):
                out[(name, str(stage))] = fn
    return out


def _sa2gfm_pretrain(_cfg: dict[str, Any]) -> None:
    from pygfm.cli.sa2gfm import pretrain_main

    pretrain_main()


def _sa2gfm_downstream(_cfg: dict[str, Any]) -> None:
    from pygfm.cli.sa2gfm import downstream_main

    downstream_main()


def _build_registry() -> dict[tuple[str, str], Runner]:
    reg: dict[tuple[str, str], Runner] = {}

    for b, s in ALL_SCRIPT_PAIRS:
        reg[(b, s)] = make_runner(b, s)

    reg[("sa2gfm", "pretrain")] = _sa2gfm_pretrain
    reg[("sa2gfm", "downstream")] = _sa2gfm_downstream

    reg.update(_discover_module_runners())

    # Always prefer in-process MDGPT runners (no scripts/ / no wheel omissions). Overrides make_runner.
    try:
        from pygfm.cli.mdgpt_stages import run_mdgpt_finetune, run_mdgpt_pretrain
    except ImportError:
        pass
    else:
        reg[("mdgpt", "pretrain")] = run_mdgpt_pretrain
        reg[("mdgpt", "finetune")] = run_mdgpt_finetune

    return reg


_REGISTRY: dict[tuple[str, str], Runner] | None = None


def get_registry() -> dict[tuple[str, str], Runner]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def run_from_yaml_dict(cfg: dict[str, Any]) -> None:
    merged = merge_config(cfg)
    baseline = merged.get("baseline")
    stage = merged.get("stage")
    if baseline is None or stage is None:
        raise ValueError("YAML must set `baseline` and `stage`.")

    b = str(baseline).strip().lower()
    s = str(stage).strip().lower()
    s = _STAGE_ALIASES.get(s, s)

    fn = get_registry().get((b, s))
    if fn is None:
        raise ValueError(
            f"No runner for baseline={b!r}, stage={s!r}. "
            f"Known: {len(list_implemented())} pairs — see list_implemented()."
        )
    fn(merged)


def list_implemented() -> list[str]:
    return [f"{a}/{t}" for (a, t) in sorted(get_registry().keys())]
