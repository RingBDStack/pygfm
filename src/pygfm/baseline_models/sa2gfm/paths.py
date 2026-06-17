"""
SA²GFM data layout (integrated in GFM-Toolbox).

Override with environment variable:
  SA2GFM_DATA_ROOT — directory containing ``ori/`` or flat ``*.pt``, ``communities/``, ``few_shot/``, …

Default (no env): prefer ``datasets/sa2gfm/data/ori/*.pt``; else if ``datasets/sa2gfm/*.pt`` is flat,
root is ``datasets/sa2gfm``; else ``datasets/sa2gfm/data`` or in-package ``data/``.

Full layout table: ``docs/sa2gfm/PATHS.md``.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_toolbox_sa2gfm_data_root(project_root: Path) -> Path:
    """
    Pick SA2GFM data_root under the repo (same rules as ``scripts/sa2gfm/_setup_repo.py``).
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


class Paths:
    """Resolved once at import; safe to read from any script."""

    def __init__(self) -> None:
        self.sa2gfm_package_root: Path = Path(__file__).resolve().parent
        self.attack_gen_root: Path = self.sa2gfm_package_root / "attack_data_gen"

        env_root = os.environ.get("SA2GFM_DATA_ROOT", "").strip()
        if env_root:
            self.data_root = Path(env_root).expanduser().resolve()
        else:
            project_root = self.sa2gfm_package_root.parents[2]
            sa_dir = project_root / "datasets" / "sa2gfm"
            if sa_dir.exists():
                self.data_root = resolve_toolbox_sa2gfm_data_root(project_root)
            else:
                self.data_root = (self.sa2gfm_package_root / "data").resolve()

        ori_sub = self.data_root / "ori"
        if ori_sub.is_dir():
            self.graph_ori_dir = ori_sub
        else:
            self.graph_ori_dir = self.data_root

        self.communities_dir = self.data_root / "communities"
        self.save_model_dir = self.data_root / "save_model"
        self.save_model_many_dir = self.data_root / "save_model_many"
        self.few_shot_dir = self.data_root / "few_shot"
        self.reduced_embeddings_dir = self.data_root / "reduced_embeddings"
        self.checkpoints_dir = self.attack_gen_root / "checkpoints"
        self.output_root = self.attack_gen_root / "outputs"
        self.attack_post_dir = self.output_root / "attack_post"
        self.attack_random_dir = self.output_root / "attacked_data_random"
        self.surrogate_deeprobust_dir = self.output_root / "surrogate_deeprobust"
        self.metattack_batch_dir = self.output_root / "metattack_batch"

    def resolve_ori_graph_pt(self, dataset: str) -> Path:
        """
        Resolve ``{dataset}.pt``: search ``graph_ori_dir``, ``data_root``, and parent of
        ``.../sa2gfm/data`` i.e. ``.../sa2gfm`` (flat Cora.pt); filenames are case-insensitive (``cora`` ↔ ``Cora.pt``).
        """
        want = dataset.strip()
        if not want:
            raise ValueError("empty dataset name")

        roots: list[Path] = []
        seen: set[str] = set()

        def add_root(p: Path) -> None:
            if not p.is_dir():
                return
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                roots.append(p.resolve())

        add_root(self.graph_ori_dir)
        add_root(self.data_root)
        if self.data_root.name == "data":
            par = self.data_root.parent
            if par.name == "sa2gfm":
                add_root(par)

        want_cf = want.casefold()
        for root in roots:
            exact = root / f"{want}.pt"
            if exact.is_file():
                return exact
            try:
                for p in root.iterdir():
                    if not p.is_file() or p.suffix.lower() != ".pt":
                        continue
                    if p.stem.casefold() == want_cf:
                        return p
            except OSError:
                continue

        raise FileNotFoundError(
            f"SA²GFM: no graph .pt for dataset {want!r}. Searched directories: {roots}. "
            f"Put e.g. Cora.pt under SA2GFM_DATA_ROOT or under .../sa2gfm/ next to data/. "
            f"Current data_root={self.data_root}"
        )

    def ensure_output_dirs(self) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.attack_post_dir.mkdir(parents=True, exist_ok=True)
        self.attack_random_dir.mkdir(parents=True, exist_ok=True)
        self.surrogate_deeprobust_dir.mkdir(parents=True, exist_ok=True)
        self.metattack_batch_dir.mkdir(parents=True, exist_ok=True)


paths = Paths()


def reinit_paths() -> None:
    """Rebuild global ``paths`` after ``SA2GFM_DATA_ROOT`` is set (e.g. by ``scripts/sa2gfm/_setup_repo.py``)."""
    global paths
    paths = Paths()
