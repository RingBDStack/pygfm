"""Resolve ``(baseline, stage)`` → runner for ``pygfm -c config.yaml``."""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import tempfile
from typing import Any, Callable

from pygfm.cli.baselines.stub_config import ALL_SCRIPT_PAIRS
from pygfm.cli.script_runner import (
    _flatten_cfg_for_stage,
    _sa2gfm_apply_data_root_from_flat,
    _yaml_dump,
    make_runner,
)

Runner = Callable[[dict[str, Any]], None]

_STAGE_ALIASES: dict[str, str] = {
    "ft": "finetune",
    "finetune_node": "finetune",
    "train": "pretrain",
}

# YAML / docs sometimes use folder-style names; registry keys follow ``scripts/<name>/``.
_BASELINE_ALIASES: dict[str, str] = {
    "rag_g_fm": "rag_gfm",
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


def _run_sa2gfm_inprocess_from_run_yaml(
    cfg: dict[str, Any], *, stage: str, main: Callable[[], None]
) -> None:
    """
    ``run_yaml`` passes a nested HF-style dict (``pretrain:`` / ``downstream:`` blocks, ``data_root``).
    In-process SA²GFM CLIs read ``-c`` via ``sys.argv`` only; they never receive ``cfg`` as an argument.
    Flatten, apply ``SA2GFM_DATA_ROOT``, write a temp YAML, then temporarily point ``argv`` at it.
    """
    flat = dict(_flatten_cfg_for_stage(cfg, stage))
    _sa2gfm_apply_data_root_from_flat(flat)
    tmp_path: str | None = None
    old_argv = sys.argv[:]
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix=f"pygfm_sa2gfm_{stage}_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_yaml_dump(flat))
        prog = old_argv[0] if old_argv else sys.executable
        sys.argv = [prog, "-c", tmp_path]
        main()
    finally:
        sys.argv = old_argv
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _sa2gfm_pretrain(cfg: dict[str, Any]) -> None:
    from pygfm.cli.sa2gfm import pretrain_main

    _run_sa2gfm_inprocess_from_run_yaml(cfg, stage="pretrain", main=pretrain_main)


def _sa2gfm_downstream(cfg: dict[str, Any]) -> None:
    from pygfm.cli.sa2gfm import downstream_main

    _run_sa2gfm_inprocess_from_run_yaml(cfg, stage="downstream", main=downstream_main)


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

    # ``stage: downstream`` = few-shot / graph-batch split generation (HF-style YAML), all prompt baselines
    # that ship ``scripts/<name>/generate_downstream.py`` share one in-process runner (scripts lack ``-c``).
    from pygfm.cli.downstream_split_runner import (
        DOWNSTREAM_SPLIT_YAML_BASELINES,
        run_downstream_splits_for_yaml,
    )

    for split_b in DOWNSTREAM_SPLIT_YAML_BASELINES:
        if split_b == "graphprompt":
            continue
        reg[(split_b, "downstream")] = run_downstream_splits_for_yaml

    try:
        from pygfm.cli.graphkeeper_stages import run_graphkeeper_domain_il
    except ImportError:
        pass
    else:
        reg[("graphkeeper", "domain_il")] = run_graphkeeper_domain_il

    try:
        from pygfm.cli.baselines.graphprompt_gc import RUNNERS as _GP_GC_RUNNERS

        for stage, fn in _GP_GC_RUNNERS.items():
            reg[("graphprompt", stage)] = fn
    except ImportError:
        pass

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
    b = _BASELINE_ALIASES.get(b, b)
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
