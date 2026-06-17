"""Add repo root to path; set default SA2GFM_DATA_ROOT."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root


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


def sa2gfm_baseline_models_root() -> Path:
    """
    ``.../site-packages/pygfm/baseline_models/sa2gfm`` (or editable ``src/pygfm/...``).

    Driver scripts must not assume ``<project>/pygfm/...`` exists; with ``pip install`` the
    package lives under ``pygfm``'s install root, not under the user's data repo.
    """
    import pygfm

    return Path(pygfm.__file__).resolve().parent / "baseline_models" / "sa2gfm"


def setup_repo() -> Path:
    """Parent of scripts/sa2gfm = repo root."""
    root = driver_script_repo_root(__file__)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    pp = os.environ.get("PYTHONPATH", "")
    sep = os.pathsep
    os.environ["PYTHONPATH"] = f"{root}{sep}{pp}" if pp else str(root)
    if not os.environ.get("SA2GFM_DATA_ROOT"):
        os.environ["SA2GFM_DATA_ROOT"] = str(_pick_sa2gfm_data_root(root))
    try:
        from pygfm.baseline_models.sa2gfm.paths import reinit_paths

        reinit_paths()
    except Exception:
        pass
    return root
