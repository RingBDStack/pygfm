"""
Canonical list of ``(baseline, stage)`` pairs dispatched by ``script_runner`` (unless a
real ``RUNNERS`` in ``pygfm.cli.baselines.<name>`` overrides).

Used by ``baseline_registry`` to register ``make_runner(b, s)`` for every pair.
"""

from __future__ import annotations

from pygfm.cli.script_runner import DEFAULT_STAGE_FILES, SCRIPT_OVERRIDES

# Folders under scripts/ with the usual pretrain / finetune / finetune_graph layout.
_STANDARD_BASELINES: tuple[str, ...] = (
    "bridge",
    "gcot",
    "graver",
    "graphkeeper",
    "graphmore",
    "graphprompt",
    "hgprompt",
    "mdgfm",
    "mdgpt",
    "rag_gfm",
    "samgpt",
)

_STANDARD_STAGES: tuple[str, ...] = ("pretrain", "finetune", "finetune_graph")


def _all_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for b in _STANDARD_BASELINES:
        for s in _STANDARD_STAGES:
            pairs.add((b, s))
    pairs.update(SCRIPT_OVERRIDES.keys())
    pairs.add(("sa2gfm", "detect"))
    return sorted(pairs)


ALL_SCRIPT_PAIRS: list[tuple[str, str]] = _all_pairs()
