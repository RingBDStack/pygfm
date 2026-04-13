"""Add repo root to path; set default SA2GFM_DATA_ROOT."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _pick_sa2gfm_data_root(project_root: Path) -> Path:
    """
    Same rules as ``pygfm.baseline_models.sa2gfm.paths.resolve_toolbox_sa2gfm_data_root``
    (duplicated here to avoid importing ``paths`` before env is set, which would construct ``Paths()`` too early).
    """
    sa = (project_root / "datasets" / "sa2gfm").resolve()
    nested = sa / "data"
    ori = nested / "ori"

    def _dir_has_pt(d: Path) -> bool:
        if not d.is_dir():
            return False
        try:
            return any(p.is_file() and p.suffix.lower() == ".pt" for p in d.iterdir())
        except OSError:
            return False

    if _dir_has_pt(ori):
        return nested
    if _dir_has_pt(sa):
        return sa
    if nested.is_dir():
        return nested
    return nested


def setup_repo() -> Path:
    """Parent of scripts/sa2gfm = repo root."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    pp = os.environ.get("PYTHONPATH", "")
    sep = os.pathsep
    os.environ["PYTHONPATH"] = f"{root}{sep}{pp}" if pp else str(root)
    if not os.environ.get("SA2GFM_DATA_ROOT"):
        os.environ["SA2GFM_DATA_ROOT"] = str(_pick_sa2gfm_data_root(root))
    return root
