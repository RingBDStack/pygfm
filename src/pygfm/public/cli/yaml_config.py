"""
Merge YAML defaults into argparse (used with ``-c`` / ``--config``).

- ``parse_args_with_optional_yaml``: lightweight merge (no warning on unknown keys).
- ``parse_args_with_config``: MDGPT-style scripts; supports ``--export-*-yaml``; warns on unknown keys.

Requires: ``pip install pyyaml``
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

from pygfm.public.cli.export_yaml import (
    EXPORT_ARG_DESTS,
    add_export_yaml_arguments,
    handle_export_args,
)


def parse_args_with_optional_yaml(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
):
    if argv is None:
        argv = sys.argv[1:]
    try:
        import yaml
    except ImportError:
        return parser.parse_args(argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=None, metavar="PATH")
    known, rest = pre.parse_known_args(argv)
    if known.config:
        path = Path(known.config).expanduser().resolve()
        if not path.is_file():
            parser.error(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = raw if isinstance(raw, dict) else {}
        dests = {
            a.dest
            for a in parser._actions
            if getattr(a, "dest", None) not in (None, "help", "config")
        }
        filtered = {k: v for k, v in data.items() if k in dests}
        parser.set_defaults(**filtered)
    return parser.parse_args(rest)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Reading YAML requires PyYAML: pip install pyyaml") from e
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return raw


def _known_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {a.dest for a in parser._actions if getattr(a, "dest", None) and a.dest != "help"}


def merge_yaml_defaults(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    ignore_keys: Optional[set[str]] = None,
) -> None:
    """Apply YAML key/value pairs as ``parser.set_defaults`` for registered dests only."""
    known = _known_dests(parser)
    merged: dict[str, Any] = {}
    skip = EXPORT_ARG_DESTS | {"config"}
    ign = frozenset(ignore_keys) if ignore_keys else frozenset()
    for k, v in config.items():
        nk = str(k).replace("-", "_")
        if nk.startswith("_") or nk in skip or nk in ign:
            continue
        if nk not in known:
            warnings.warn(f"YAML key not registered on this script, ignored: {k!r}", stacklevel=2)
            continue
        if v is None:
            continue
        action = next((a for a in parser._actions if a.dest == nk), None)
        if action is None:
            continue
        act = getattr(action, "action", None)
        if action.nargs in ("+", "*") or action.nargs is not None and isinstance(action.nargs, int):
            merged[nk] = v
            continue
        if act in ("store_true", "store_false"):
            if isinstance(v, bool):
                merged[nk] = v
            else:
                merged[nk] = bool(v)
            continue
        merged[nk] = v
    if merged:
        parser.set_defaults(**merged)


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[list[str]] = None,
    *,
    script_file: Path | None = None,
) -> argparse.Namespace:
    """
    Parse ``-c`` / ``--config``, merge YAML into defaults, then parse remaining CLI (CLI wins).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", type=str, default=None, dest="config")
    pre_args, rest = pre.parse_known_args(argv)

    if not any(getattr(a, "dest", None) == "config" for a in parser._actions):
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            dest="config",
            help="YAML config path (overrides code defaults; overridden by CLI)",
        )

    add_export_yaml_arguments(parser)

    cfg: dict[str, Any] = {}
    if pre_args.config:
        cfg = load_yaml(pre_args.config)
    merge_yaml_defaults(parser, cfg)

    args = parser.parse_args(rest)
    if getattr(args, "config", None) is None and pre_args.config is not None:
        args.config = pre_args.config
    handle_export_args(parser, args, rest, script_file=script_file)
    return args


def ensure_config_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``-c`` / ``--config`` if missing."""
    if not any(getattr(a, "dest", None) == "config" for a in parser._actions):
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            dest="config",
            help="YAML config path (overrides code defaults; overridden by CLI)",
        )
