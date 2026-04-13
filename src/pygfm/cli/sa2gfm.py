"""SA²GFM: stable console entrypoints (no dependency on ``scripts/`` layout)."""

from __future__ import annotations


def pretrain_main() -> None:
    from pygfm.baseline_models.sa2gfm.pretrain.pipeline.train_single import train

    train()


def downstream_main() -> None:
    from pygfm.baseline_models.sa2gfm.downstream.pipeline.train_downstream import main

    main()
