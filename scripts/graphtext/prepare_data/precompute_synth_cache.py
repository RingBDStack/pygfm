#!/usr/bin/env python3
"""
Precompute GraphText synth-related cache artifacts without running LLM inference.

This script only builds dataset-side caches (e.g., SPD/PPR/proxy graphs and text fields
derived from graph propagation) by initializing TextualGraph with selected config overrides.

Example:
  cd /root/autodl-tmp/gfm-toolbox-main
  python scripts/graphtext/prepare_data/precompute_synth_cache.py \
    --datasets wisconsin cornell \
    --text-info a2y_t.a3y_t \
    --rel-info spd0.ppr \
    --report reports/graphtext_synth_precompute.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../gfm-toolbox-main
GRAPH_TEXT_ROOT = PROJECT_ROOT / "pygfm" / "models" / "graphtext"
CONFIG_DIR = PROJECT_ROOT / "scripts" / "graphtext" / "config"

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(GRAPH_TEXT_ROOT))

from utils.basics import init_env_variables  # noqa: E402
from utils.data.textual_graph import TextualGraph  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute GraphText synth caches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wisconsin", "cornell"],
        help="Hydra data config names, e.g. wisconsin cornell",
    )
    parser.add_argument(
        "--text-info",
        default="a2y_t.a3y_t",
        help="Hydra override for text_info, e.g. a2y_t.a3y_t",
    )
    parser.add_argument(
        "--rel-info",
        default="spd0.ppr",
        help="Hydra override for rel_info, e.g. spd0.ppr",
    )
    parser.add_argument(
        "--report",
        default="reports/graphtext_synth_precompute.json",
        help="Output report path relative to project root or absolute path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_env_variables()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        for ds in args.datasets:
            overrides = [
                f"data={ds}",
                f"text_info={args.text_info}",
                f"rel_info={args.rel_info}",
                "mode=icl",
                "debug=true",
            ]
            item = {"dataset": ds, "overrides": overrides, "status": "ok", "error": ""}
            try:
                cfg = compose(config_name="main", overrides=overrides)
                _ = TextualGraph(cfg=cfg)
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)
            results.append(item)

    summary = {
        "project_root": str(PROJECT_ROOT),
        "config_dir": str(CONFIG_DIR),
        "text_info": args.text_info,
        "rel_info": args.rel_info,
        "datasets": args.datasets,
        "results": results,
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Report written to: {report_path}")
    for item in results:
        if item["status"] == "ok":
            print(f"- {item['dataset']}: OK")
        else:
            print(f"- {item['dataset']}: FAILED - {item['error']}")


if __name__ == "__main__":
    main()
