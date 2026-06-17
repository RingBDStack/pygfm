"""
Flat YAML mapping to HuggingFace/argparse --key value argv.

Keys use underscores (HfArgumentParser style).

``run_yaml`` may merge top-level layout keys (``baseline``, ``stage``, ``data_root``, …) into the
same flat file passed to LLaGA / GraphGPT drivers. Those are skipped here; ``data_root`` is
written to ``LLAGA_DATA_ROOT`` for :mod:`pygfm.baseline_models.llaga.paths` instead of ``--data_root``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pygfm.public.cli.yaml_config import load_yaml

# Never forwarded as ``--flag`` to HuggingFace parsers (GFM run_yaml / driver YAML only).
_LAYOUT_SKIP_KEYS = frozenset({"baseline", "stage", "params"})


def _scalar_to_str(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def _argv_scalar_for_key(key: str, v: Any) -> str:
    """
    HuggingFace ``*Strategy`` fields are enums (``no``, ``steps``, ``epoch``, …).

    YAML 1.1 parses bare ``no`` / ``off`` as boolean ``False``; ``yaml_flat_to_argv`` would then emit
    ``False``, which ``HfArgumentParser`` rejects. Map bools back to the string token ``no``.
    """
    if key == "eval_strategy" and isinstance(v, bool):
        return "no" if not v else "epoch"
    if key == "save_strategy" and isinstance(v, bool):
        return "no" if not v else "steps"
    if key == "logging_strategy" and isinstance(v, bool):
        return "no" if not v else "steps"
    return _scalar_to_str(v)


def _apply_llaga_data_root_from_yaml_value(v: Any) -> None:
    """Resolve ``data_root`` relative to cwd and set ``LLAGA_DATA_ROOT`` (see ``llaga.paths``)."""
    if v is None:
        return
    s = str(v).strip()
    if not s or s.lower() in ("null", "none"):
        return
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    os.environ["LLAGA_DATA_ROOT"] = str(p)


def yaml_flat_to_argv(path: str | Path, *, skip_prefix: str = "_") -> list[str]:
    data = load_yaml(path)
    out: list[str] = []
    for k in sorted(data.keys(), key=str):
        sk = str(k)
        if sk.startswith(skip_prefix) or sk in _LAYOUT_SKIP_KEYS:
            continue
        v = data[k]
        if v is None:
            continue
        if sk == "data_root":
            _apply_llaga_data_root_from_yaml_value(v)
            continue
        out.append(f"--{sk}")
        out.append(_argv_scalar_for_key(sk, v))
    return out
