"""
Baseline finetune scripts: when ``--ckpt`` is omitted (or YAML null), probe under ``ckpts/<baseline>/``.
"""
from __future__ import annotations

from pathlib import Path


def resolve_preprompt_ckpt(
    repo_root: Path,
    baseline_dir: str,
    dataset: str,
    explicit: str | None,
    *,
    save_basename: str = "preprompt.pth",
) -> str:
    if explicit is not None and str(explicit).strip():
        raw = explicit.strip()
        p = Path(raw)
        if not p.is_file():
            alt = repo_root / raw
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError(
                    f"--ckpt path not found: {Path(raw).resolve()} (also tried repo root {alt})"
                )
        return str(p.resolve())

    ds = (dataset or "Cora").strip().lower()
    base = repo_root / "ckpts" / baseline_dir
    candidates = [
        base / save_basename,
        base / ds / f"preprompt_{ds}.pth",
        base / ds / save_basename,
    ]
    for c in candidates:
        if c.is_file():
            print(f">> --ckpt not set, using: {c}")
            return str(c.resolve())
    raise FileNotFoundError(
        f"No PrePrompt checkpoint for {baseline_dir} (dataset={dataset!r}). Pass --ckpt.\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )
