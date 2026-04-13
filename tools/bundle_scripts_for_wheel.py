"""Pack ``scripts/`` into ``src/pygfm/_scripts_bundle.zip`` (excludes ckpts, *.pth, junk). Run before release."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "src" / "pygfm" / "_scripts_bundle.zip"

SKIP_SUFFIX = {".pth"}
SKIP_DIR_NAMES = {"ckpts", "__pycache__", ".git"}


def _skip(path: Path, rel: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIX:
        return True
    if path.name in {"nohup.out", ".DS_Store"}:
        return True
    return False


def main() -> None:
    if not SCRIPTS.is_dir():
        raise SystemExit(f"Missing {SCRIPTS}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SCRIPTS.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(SCRIPTS)
            if _skip(f, rel):
                continue
            # Prefix with scripts/ so extracted layout matches repo: .../scripts/mdgpt/pretrain.py
            zf.write(f, f"scripts/{rel.as_posix()}")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
