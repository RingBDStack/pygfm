"""
Flat YAML mapping to HuggingFace/argparse --key value argv.

Keys use underscores (HfArgumentParser style).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pygfm.public.cli.yaml_config import load_yaml


def _scalar_to_str(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def yaml_flat_to_argv(path: str | Path, *, skip_prefix: str = "_") -> list[str]:
    data = load_yaml(path)
    out: list[str] = []
    for k in sorted(data.keys(), key=str):
        sk = str(k)
        if sk.startswith(skip_prefix):
            continue
        v = data[k]
        if v is None:
            continue
        out.append(f"--{sk}")
        out.append(_scalar_to_str(v))
    return out
