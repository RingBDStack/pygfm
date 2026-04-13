# LLaGA

Unified entrypoint: **`run.py`** (subcommands). DeepSpeed still uses `zero*.json` in this folder.

## Install

```bash
cd /path/to/repo
pip install -e .
# Optional extras if you maintain a local [llaga] extra in pyproject — otherwise install torch/transformers manually.
```

## Data

Place upstream `dataset/` under `datasets/llaga/dataset/` or set `LLAGA_DATA_ROOT`. See `datasets/llaga/README.md` if present.

## Training (single GPU)

```bash
export LLAGA_REPORT_TO=none   # optional: skip wandb
python scripts/llaga/run.py train --model opt_2.7b --task nc --dataset cora --bs 4 --emb simteg --max_steps 3
```

## Smoke test

Uses bundled or local OPT stub + Cora (default `--max-steps 3`):

```bash
python scripts/llaga/run.py smoke --max-steps 3
```

## Flat YAML → training

Templates: `configs/llaga/smoke.yaml` (offline smoke), `configs/llaga/train_template.yaml` (full run shape).

```bash
python scripts/llaga/run.py yaml -c configs/llaga/smoke.yaml
python scripts/llaga/run.py yaml --export-run-yaml configs/llaga/_dump.yaml -c configs/llaga/train_template.yaml
```

## Module entry (after pip install)

```bash
python -m pygfm.baseline_models.llaga.train.train_mem --help
```

## Download base models

```bash
python scripts/llaga/run.py download opt
# or: python scripts/llaga/run.py download vicuna
```

Environment variables: `LLAGA_DATA_ROOT`, `LLAGA_HF_CACHE`, `LLAGA_CKPT_ROOT`, `LLAGA_MODEL_BASE_OVERRIDE`, `LLAGA_REPORT_TO`, etc.
