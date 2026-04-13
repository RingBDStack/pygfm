"""Unified CLI: run experiments from YAML for any registered ``baseline`` + ``stage``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from pygfm.cli.baseline_registry import list_implemented, run_from_yaml_dict


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run GFM from YAML with `baseline` and `stage` (e.g. mdgpt + pretrain). "
            "Registered pairs include all stub stages from stub_config plus real RUNNERS in "
            "pygfm.cli.baselines.<name> — see list_implemented() in baseline_registry."
        ),
    )
    p.add_argument(
        "-c",
        "--config",
        "--yaml-config",
        dest="config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = p.parse_args(argv)

    path = Path(args.config)
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")

    raw = path.read_text(encoding="utf-8")
    cfg: dict[str, Any] = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise SystemExit("YAML root must be a mapping (object)")

    run_from_yaml_dict(cfg)


if __name__ == "__main__":
    main()
