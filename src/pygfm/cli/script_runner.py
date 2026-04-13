"""
Run repository ``scripts/<baseline>/*.py`` like the original CLI: write a flat YAML and pass
``-c`` (for parsers that support it), then call ``main()``.

Resolution order for ``scripts/``:

1. ``PYGFM_REPO_ROOT`` / ``PYGFM_SCRIPTS_ROOT``
2. A git checkout (walk parents from this file, or ``cwd`` with ``./scripts``)
3. **Wheel / pip install:** ``pygfm._scripts_bundle.zip`` next to this package — unpacked once under
   the system temp dir (same layout as repo ``scripts/``).

Some scripts only expose logic under ``if __name__ == "__main__"``; those need one-off handling
(see ``_run_sa2gfm_script``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

import yaml

SCRIPT_OVERRIDES: dict[tuple[str, str], str] = {
    ("graphgpt", "pretrain"): "graphgpt/run_with_config.py",
    ("graphgpt", "finetune"): "graphgpt/run_with_config.py",
    ("graphgpt", "finetune_graph"): "graphgpt/run_with_config.py",
    ("graphtext", "pretrain"): "graphtext/run_with_config.py",
    ("graphtext", "finetune"): "graphtext/run_sft.py",
    ("graphtext", "finetune_graph"): "graphtext/run_icl.py",
    ("llaga", "pretrain"): "llaga/run.py",
    ("llaga", "finetune"): "llaga/run.py",
    ("llaga", "finetune_graph"): "llaga/run.py",
    ("multigprompt", "pretrain"): "multigprompt/execute.py",
    ("multigprompt", "finetune"): "multigprompt/execute.py",
    ("multigprompt", "finetune_graph"): "multigprompt/execute.py",
    ("oneforall", "pretrain"): "oneforall/run.py",
    ("oneforall", "finetune"): "oneforall/run.py",
    ("oneforall", "finetune_graph"): "oneforall/run.py",
    ("sa2gfm", "detect"): "sa2gfm/detect.py",
}

ARGV_PREFIX: dict[tuple[str, str], list[str]] = {
    ("llaga", "pretrain"): ["yaml"],
    ("llaga", "finetune"): ["yaml"],
    ("llaga", "finetune_graph"): ["yaml"],
}

DEFAULT_STAGE_FILES: dict[str, str] = {
    "pretrain": "pretrain.py",
    "finetune": "finetune.py",
    "finetune_graph": "finetune_graph.py",
    "downstream": "downstream.py",
    "detect": "detect.py",
}


def _bundled_scripts_dir() -> Path | None:
    """Unpack ``_scripts_bundle.zip`` (shipped in the wheel) to a temp dir; return ``.../scripts``."""
    try:
        import pygfm
    except ImportError:
        return None
    pkg_root = Path(pygfm.__file__).resolve().parent
    zpath = pkg_root / "_scripts_bundle.zip"
    if not zpath.is_file():
        return None
    key = hashlib.sha256(zpath.read_bytes()).hexdigest()[:24]
    cache = Path(tempfile.gettempdir()) / "pygfm_bundled_scripts" / key
    scripts_dir = cache / "scripts"
    done = cache / ".extracted"
    if not done.is_file():
        import shutil

        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(cache)
        done.write_text("1", encoding="ascii")
    if not (scripts_dir / "mdgpt").is_dir():
        return None
    return scripts_dir.resolve()


def find_scripts_root() -> Path | None:
    env = os.environ.get("PYGFM_REPO_ROOT") or os.environ.get("PYGFM_SCRIPTS_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "scripts").is_dir():
            return (p / "scripts").resolve()
        if p.name == "scripts" and p.is_dir():
            return p.resolve()
    here = Path(__file__).resolve()
    for depth in (3, 4, 5, 6):
        if depth >= len(here.parents):
            continue
        cand = here.parents[depth] / "scripts"
        if cand.is_dir():
            return cand.resolve()
    cwd = Path.cwd()
    if (cwd / "scripts").is_dir():
        return (cwd / "scripts").resolve()
    bundled = _bundled_scripts_dir()
    if bundled is not None:
        return bundled
    return None


def resolve_script_path(baseline: str, stage: str) -> Path | None:
    b = baseline.strip().lower()
    s = stage.strip().lower()
    rel = SCRIPT_OVERRIDES.get((b, s))
    if rel is None:
        fname = DEFAULT_STAGE_FILES.get(s)
        if fname is None:
            return None
        rel = f"{b}/{fname}"
    root = find_scripts_root()
    if root is None:
        return None
    path = (root / rel.replace("/", os.sep)).resolve()
    return path if path.is_file() else None


def _flatten_cfg_for_stage(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    block_key = {
        "pretrain": "pretrain",
        "finetune": "finetune",
        "finetune_graph": "finetune_graph",
        "downstream": "downstream",
        "detect": "detect",
    }.get(stage, "pretrain")
    block = cfg.get(block_key)
    if not isinstance(block, dict):
        block = {}
    skip = {
        "baseline",
        "stage",
        "params",
        "pretrain",
        "finetune",
        "finetune_graph",
        "downstream",
        "detect",
    }
    top = {k: v for k, v in cfg.items() if k not in skip}
    merged: dict[str, Any] = {**block, **top}
    for k in list(merged.keys()):
        if k in skip:
            merged.pop(k, None)
    return merged


def _yaml_dump(data: dict[str, Any]) -> str:
    def conv(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o

    return yaml.safe_dump(conv(data), sort_keys=False, allow_unicode=True)


def _run_sa2gfm_detect(path: Path, cfg: dict[str, Any]) -> None:
    """``detect.py`` uses ``_main()`` only under ``if __name__``."""
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import importlib

    setup = importlib.import_module("_setup_repo")
    setup.setup_repo()
    spec = importlib.util.spec_from_file_location("sa2gfm_detect", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "_main"):
        mod._main()
    else:
        raise RuntimeError(f"No _main in {path}")


def run_script_main(baseline: str, stage: str, cfg: dict[str, Any]) -> None:
    path = resolve_script_path(baseline, stage)
    if path is None:
        raise FileNotFoundError(
            f"No script for baseline={baseline!r}, stage={stage!r}. "
            "Set PYGFM_REPO_ROOT to the repo root, run from a checkout with scripts/, "
            "or use a python-gfm wheel that includes _scripts_bundle.zip."
        )

    if baseline.lower() == "sa2gfm" and stage.lower() == "detect":
        _run_sa2gfm_detect(path, cfg)
        return

    flat = _flatten_cfg_for_stage(cfg, stage)
    tmp_path: str | None = None
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    repo_root = path.resolve().parents[2]
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix=f"pygfm_{baseline}_{stage}_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_yaml_dump(flat))

        prefix = ARGV_PREFIX.get((baseline.lower(), stage.lower()), [])
        sys.argv = [str(path.resolve())] + prefix + ["-c", tmp_path]
        os.chdir(repo_root)

        spec = importlib.util.spec_from_file_location(
            f"_pygfm_{baseline}_{stage}_{path.stem}",
            path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        main_fn = getattr(mod, "main", None)
        if callable(main_fn):
            main_fn()
        elif hasattr(mod, "_main") and callable(mod._main):
            mod._main()
        else:
            raise RuntimeError(
                f"{path} has no main() — this baseline may need a custom runner in pygfm.cli.baseline_registry."
            )
    finally:
        sys.argv = old_argv
        try:
            os.chdir(old_cwd)
        except OSError:
            pass
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def make_runner(baseline: str, stage: str) -> Callable[[dict[str, Any]], None]:
    def _run(cfg: dict[str, Any]) -> None:
        run_script_main(baseline, stage, cfg)

    return _run
