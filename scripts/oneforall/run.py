#!/usr/bin/env python3
"""
OneForAll: ``-c`` / ``--config`` YAML merged over packaged ``default_config.yaml``; supports ``--export-run-yaml``.

Config path resolution:

- Absolute paths work from anywhere.
- Relative paths: resolved from the **current working directory** first; if missing, from the **repository root**
  (so ``configs/oneforall/smoke.yaml`` or ``pygfm/baseline_models/oneforall/configs/task_config.yaml`` work).

Trailing ``key value ...`` tokens are forwarded as ``merge_mod`` overrides (same as ``run_cdm`` CLI opts).

Examples::

    python scripts/oneforall/run.py -c configs/oneforall/smoke.yaml
    python scripts/oneforall/run.py -c pygfm/baseline_models/oneforall/configs/task_config.yaml
    python scripts/oneforall/run.py --export-run-yaml /tmp/merged.yaml -c configs/oneforall/smoke.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

import yaml

_ROOT = driver_script_repo_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pygfm.baseline_models.oneforall.gp.utils.utils import (
    combine_dict,
    load_yaml,
    merge_mod,
)
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, resolve_export_path


def _default_yaml() -> Path:
    """Packaged default lives under ``pygfm.baseline_models.oneforall``; cwd-based ``_ROOT`` may not contain ``pygfm/``."""
    tail = Path("pygfm") / "baseline_models" / "oneforall" / "configs" / "default_config.yaml"
    candidates = [
        _ROOT / tail,
        _ROOT / "src" / tail,
    ]
    try:
        import pygfm.baseline_models.oneforall as _ofa

        candidates.append(Path(_ofa.__file__).resolve().parent / "configs" / "default_config.yaml")
    except ImportError:
        pass
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "OneForAll default_config.yaml not found. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def _resolve_config_path(p: str) -> str:
    """Resolve user-supplied config path: cwd first, then repo root."""
    path = Path(p)
    if path.is_file():
        return str(path.resolve())
    rooted = (_ROOT / path).resolve()
    if rooted.is_file():
        return str(rooted)
    raise FileNotFoundError(
        f"Config not found: {p!r} (tried {path.resolve()} and {rooted})"
    )


def _merged_params(config_path: str | None, mod_tokens: list[str]) -> dict:
    stacks = [load_yaml(str(_default_yaml()))]
    if config_path:
        stacks.append(load_yaml(_resolve_config_path(config_path)))
    p = combine_dict(*stacks)
    return merge_mod(p, mod_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="OneForAll YAML entry (GFM-Toolbox)")
    parser.add_argument("-c", "--config", type=str, default=None, metavar="PATH")
    add_export_yaml_arguments(parser)
    parser.add_argument(
        "opts",
        nargs="*",
        default=[],
        help="Append key value key value ... (written to merge_mod)",
    )
    args = parser.parse_args()
    script_file = Path(__file__)

    if getattr(args, "export_default_yaml", None):
        out = resolve_export_path(args.export_default_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = load_yaml(str(_default_yaml()))
        out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote default template -> {out}")
        return

    merged = _merged_params(args.config, args.opts)

    if getattr(args, "export_run_yaml", None):
        out = resolve_export_path(args.export_run_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote merged run config -> {out}")
        return

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.safe_dump(merged, tmp, sort_keys=False, allow_unicode=True)
        tmp_path = tmp.name
    cmd = [
        sys.executable,
        "-m",
        "pygfm.baseline_models.oneforall.run_cdm",
        "--override",
        tmp_path,
    ]
    print(">>", " ".join(cmd))
    try:
        raise SystemExit(subprocess.call(cmd, cwd=_ROOT))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
