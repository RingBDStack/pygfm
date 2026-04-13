# SA²GFM (GFM-Toolbox)

## Install

```bash
cd /path/to/repo
pip install -e .
# Optional attack stack (Nettack / Metattack, etc.) may need extra deps — see docs/sa2gfm.
```

### PyPI-style console commands

After `pip install` (wheel or editable), you can run the packaged entrypoints from any working directory (data paths still follow `SA2GFM_DATA_ROOT` / `datasets/sa2gfm`):

```bash
gfm-sa2gfm-pretrain -c configs/sa2gfm/pretrain.yaml
gfm-sa2gfm-downstream -c configs/sa2gfm/downstream.yaml
```

Equivalent from a full git checkout at repo root:

```bash
python scripts/sa2gfm/pretrain.py -c configs/sa2gfm/pretrain.yaml
python scripts/sa2gfm/downstream.py -c configs/sa2gfm/downstream.yaml
```

## Data layout

Default roots: if `datasets/sa2gfm/data/ori/*.pt` exists, use that tree; otherwise flat `datasets/sa2gfm/*.pt` works (common GFM export layout). Override with:

```bash
export SA2GFM_DATA_ROOT=/path/to/data
```

More detail: [docs/sa2gfm/README.md](../../docs/sa2gfm/README.md).

## Scripts (run from repository root)

Most steps accept **`-c` / `--config`** YAML plus **`--export-run-yaml`** / **`--export-default-yaml`** (needs `pyyaml`). Chained attack scripts only YAML-wrap their own flags; inner `pipeline/*.py` modules may be called directly with `-c`.

| Script | Role |
|--------|------|
| `pretrain.py` | Single-graph contrastive pretrain |
| `pretrain_experts_for_downstream.py` | Sequentially pretrain **other-graph** experts for MoE |
| `downstream.py` | MoE downstream finetuning |
| `generate_fewshot.py` | Few-shot split files |
| `detect.py` | Community detection (single dataset) |
| `attack_train_surrogate.py` | Train GCN surrogate for Nettack |
| `attack_targeted_nettack.py` | Pipeline: ① (optional) → ② Nettack → ③ merge `.pt` |
| `attack_random.py` | Random feature/structure attack |
| `attack_metattack.py` | ⑤ surrogate → ⑥ Metattack batch (extra deps) |

Set **`ATTACK_DEVICE=cpu`** for CPU-heavy Metattack stages. Skip surrogate training with **`SKIP_SURROGATE_TRAIN=1`** or **`--skip-surrogate-train`**.

Examples:

```bash
python scripts/sa2gfm/pretrain.py --dataset cora --nb_epochs 1 --no_swanlab
python scripts/sa2gfm/downstream.py --dataset cora --shot_num 1 --num-splits 1 --no-swanlab
python scripts/sa2gfm/attack_targeted_nettack.py --dataset cora --p 1 --skip-surrogate-train
```

Batch expert pretrain (arguments after `--` are forwarded to each `pretrain.py`):

```bash
python scripts/sa2gfm/pretrain_experts_for_downstream.py --target cora --dry_run
python scripts/sa2gfm/pretrain_experts_for_downstream.py --target cora -- -c configs/sa2gfm/pretrain.yaml
```

If only `save_model/cora.pt` exists, `downstream.py` falls back to a **single-expert MoE** with a warning; use `pretrain_experts_for_downstream.py` for full MoE.

## In-package Python entry (same as scripts)

From repo root you may also run:

```bash
python pygfm/baseline_models/sa2gfm/pretrain/pipeline/train_single.py --dataset cora --nb_epochs 1 --no_swanlab
python pygfm/baseline_models/sa2gfm/downstream/pipeline/train_downstream.py --dataset cora --shot_num 1 --num-splits 1 --no-swanlab
```

Pickle checkpoints reference `pygfm.baseline_models.sa2gfm.pretrain.pipeline.model`. Older standalone-repo checkpoints that pickle a top-level `model` class may need that legacy code on `PYTHONPATH` or a fresh pretrain.
