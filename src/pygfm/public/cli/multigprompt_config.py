"""
MultiGPrompt: YAML + Pydantic; supports ``-c``, ``--export-run-yaml``, ``--export-default-yaml``.

Example: ``configs/multigprompt/example_cora.yaml``

Requires: ``pip install pydantic pyyaml``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

from pygfm.public.cli.export_yaml import add_export_yaml_arguments, dump_namespace_to_yaml

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MultiGPrompt YAML mode needs pydantic. Run: pip install pydantic pyyaml"
    ) from e

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MultiGPrompt YAML mode needs PyYAML. Run: pip install pyyaml"
    ) from e


class MultiGPromptRunConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    dataset: str = Field(default="cora")
    aug_type: str = Field(default="edge")
    drop_percent: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = Field(default=39)
    gpu: int = Field(default=0, ge=0)
    save_name: Optional[str] = Field(default=None)
    epochs: int = Field(default=1000, ge=1)
    tasks: int = Field(default=100, ge=1)
    inner_steps: int = Field(default=50, ge=1)
    no_swanlab: bool = Field(
        default=False,
        description="Match other baseline CLIs; execute script does not wire SwanLab yet",
    )
    splits_path: Optional[str] = Field(
        default=None,
        description="Absolute/relative path to splits.pt, or a directory containing it (e.g. .../1shot). "
        "If unset, search default multigprompt / mdgpt locations.",
    )


def _argv_specifies_long_opt(argv: list[str], dest: str) -> bool:
    prefix = f"--{dest}"
    return any(a == prefix or a.startswith(prefix + "=") for a in argv)


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def parse_args(
    argv: list[str] | None = None,
    *,
    script_file: Path | None = None,
) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        "MultiGPrompt (Planetoid node)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples: configs/multigprompt/example_cora.yaml; requires: pip install pydantic pyyaml",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML config (optional)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--aug_type", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--drop_percent", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--tasks", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inner_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--splits_path",
        type=str,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="Path to splits.pt or its parent directory (e.g. downstream_data/mdgpt/cora/1shot)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--no_swanlab",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    add_export_yaml_arguments(parser)

    args = parser.parse_args(argv)

    _SKIP_MERGE = frozenset(
        {"config", "smoke", "export_default_yaml", "export_run_yaml"}
    )

    if getattr(args, "export_default_yaml", None):
        base = MultiGPromptRunConfig().model_dump()
        for name in base:
            if _argv_specifies_long_opt(argv, name) and hasattr(args, name):
                base[name] = getattr(args, name)
        smoke = bool(getattr(args, "smoke", False))
        if smoke:
            if not _argv_specifies_long_opt(argv, "epochs"):
                base["epochs"] = 1
            if not _argv_specifies_long_opt(argv, "tasks"):
                base["tasks"] = 1
            if not _argv_specifies_long_opt(argv, "inner_steps"):
                base["inner_steps"] = 1
        dump_namespace_to_yaml(
            argparse.Namespace(**base),
            args.export_default_yaml,
            script_file=script_file,
        )
        sys.exit(0)

    yaml_dict: dict = {}
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        yaml_dict = _load_yaml(cfg_path)

    merged = {**MultiGPromptRunConfig().model_dump(), **yaml_dict}
    smoke = bool(getattr(args, "smoke", False))
    for k, v in vars(args).items():
        if k in _SKIP_MERGE:
            continue
        merged[k] = v

    if smoke:
        if not _argv_specifies_long_opt(argv, "epochs"):
            merged["epochs"] = 1
        if not _argv_specifies_long_opt(argv, "tasks"):
            merged["tasks"] = 1
        if not _argv_specifies_long_opt(argv, "inner_steps"):
            merged["inner_steps"] = 1

    validated = MultiGPromptRunConfig.model_validate(merged)

    if getattr(args, "export_run_yaml", None):
        dump_namespace_to_yaml(
            argparse.Namespace(**validated.model_dump()),
            args.export_run_yaml,
            script_file=script_file,
        )
        sys.exit(0)

    return argparse.Namespace(**validated.model_dump())
