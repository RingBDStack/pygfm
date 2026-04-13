"""
MDGPT scripts: load YAML defaults then argparse (precedence: code defaults < config file < CLI).

Supports ``-c`` / ``--config`` and ``--export-default-yaml`` / ``--export-run-yaml``
(bare export filenames go next to the calling script).

Requires: ``pip install pyyaml``
"""
from __future__ import annotations

from pygfm.public.cli.yaml_config import (
    load_yaml,
    merge_yaml_defaults,
    parse_args_with_config,
    ensure_config_arg,
)

__all__ = [
    "load_yaml",
    "merge_yaml_defaults",
    "parse_args_with_config",
    "ensure_config_arg",
]
