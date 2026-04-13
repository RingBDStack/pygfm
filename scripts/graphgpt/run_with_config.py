#!/usr/bin/env python3
"""
GraphGPT ``train_mem``: flat YAML keys map to ``train_graph.HfArgumentParser`` flags (``--key value``).

From repo root::

    python scripts/graphgpt/run_with_config.py -c configs/graphgpt/train_mem_smoke.yaml
    python scripts/graphgpt/run_with_config.py --export-run-yaml configs/graphgpt/_dump.yaml -c configs/graphgpt/train_mem_smoke.yaml

Single-GPU calls ``train_mem`` directly; for multi-GPU wrap with ``torchrun`` yourself.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pygfm.public.cli.export_yaml import add_export_yaml_arguments, resolve_export_path
from pygfm.public.cli.yaml_config import load_yaml
from pygfm.public.cli.yaml_flat_to_argv import yaml_flat_to_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphGPT train_mem YAML entry")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="PATH")
    add_export_yaml_arguments(parser)
    args = parser.parse_args()
    script_file = Path(__file__)

    if getattr(args, "export_default_yaml", None):
        out = resolve_export_path(args.export_default_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        tpl = {
            "_comment": "See configs/graphgpt/train_mem_smoke.yaml; keys match bash tune_script args",
            "model_name_or_path": "facebook/opt-125m",
            "output_dir": "ckpts/graphgpt/checkpoints/yaml_smoke",
            "max_steps": 1,
            "report_to": "none",
        }
        out.write_text(yaml.safe_dump(tpl, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote template -> {out}")
        return

    data = load_yaml(args.config)
    if getattr(args, "export_run_yaml", None):
        out = resolve_export_path(args.export_run_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote -> {out}")
        return

    extra = yaml_flat_to_argv(args.config)
    cmd = [
        sys.executable,
        "-m",
        "pygfm.baseline_models.graphgpt.train.train_mem",
        *extra,
    ]
    print(">>", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
