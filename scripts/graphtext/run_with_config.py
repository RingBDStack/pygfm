#!/usr/bin/env python3
"""
GraphText (Hydra): YAML sets ``entry`` and base ``hydra_overrides``; append dataset etc. on the CLI (forwarded to ``run_sft.py`` / ``run_icl.py``).

From repo root::

    python scripts/graphtext/run_with_config.py -c configs/graphtext/sft.yaml data=cora
    python scripts/graphtext/run_with_config.py -c configs/graphtext/icl.yaml data=pubmed max_test_samples=50
    python scripts/graphtext/run_with_config.py -c configs/graphtext/sft_smoke.yaml data=cora
    python scripts/graphtext/run_with_config.py -c configs/graphtext/icl_smoke.yaml data=cora
    python scripts/graphtext/run_with_config.py --export-run-yaml out.yaml -c configs/graphtext/sft.yaml
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


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphText Hydra YAML entry")
    parser.add_argument("-c", "--config", type=str, required=True, metavar="PATH")
    add_export_yaml_arguments(parser)
    args, unknown = parser.parse_known_args()
    script_file = Path(__file__)
    cfg = load_yaml(args.config)

    if getattr(args, "export_default_yaml", None):
        tpl = {
            "_comment": "GraphText: entry=sft|icl; hydra_overrides are Hydra CLI overrides",
            "entry": "sft",
            "hydra_overrides": ["exp=sft", "total_steps=700"],
        }
        out = resolve_export_path(args.export_default_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(tpl, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote template -> {out}")
        return

    entry = str(cfg.get("entry", "sft")).lower()
    overrides = list(cfg.get("hydra_overrides") or [])
    extra = list(unknown)
    while extra and extra[0] == "--":
        extra.pop(0)
    merged = overrides + extra
    if not merged:
        parser.error(
            "hydra_overrides in YAML and extra CLI overrides cannot both be empty; "
            "e.g. YAML sets exp=sft and CLI adds data=cora"
        )

    if getattr(args, "export_run_yaml", None):
        out = resolve_export_path(args.export_run_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote (copy of -c) -> {out}")
        return

    script_name = "run_sft.py" if entry in ("sft", "run_sft", "train") else "run_icl.py"
    target = _ROOT / "scripts" / "graphtext" / script_name
    if not target.is_file():
        raise FileNotFoundError(target)
    cmd = [sys.executable, str(target), *merged]
    print(">>", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
