# MDGPT (PrePrompt + DownPrompt)

## Layout

- **Pretrain**: multi-domain node contrastive learning (`PrePromptModel`).
- **Downstream**: frozen GCN + trainable input prompt + prototype matching (`DownPromptModel` / `DownPromptGraphModel`).
- Data convention: `datasets/mdgpt/` (or `GFM_DATA_ROOT`).

## Install

```bash
cd /path/to/repo
pip install -e .
```

Run experiment scripts **from the repository root** so relative paths resolve.

## Experiments

| Step | Script | YAML template |
|------|--------|----------------|
| Pretrain | `python scripts/mdgpt/pretrain.py` | `configs/mdgpt/pretrain.yaml` |
| Few-shot splits | `python scripts/mdgpt/generate_downstream.py few_shot` | `configs/mdgpt/generate_downstream.yaml` |
| Graph-level splits | `python scripts/mdgpt/generate_downstream.py graph_batch` | same (set `mode`) |
| Node k-shot finetune | `python scripts/mdgpt/finetune.py` | `configs/mdgpt/finetune.yaml` |
| Graph k-shot finetune | `python scripts/mdgpt/finetune_graph.py` | `configs/mdgpt/finetune_graph.yaml` |
| 100-task sweep | `python scripts/mdgpt/run_1shot_100task.py` | `configs/mdgpt/run_1shot_100task.yaml` |

Export merged defaults:

```bash
python scripts/mdgpt/pretrain.py --export-default-yaml configs/mdgpt/_defaults.yaml
python scripts/mdgpt/pretrain.py --export-run-yaml configs/mdgpt/_run.yaml -c configs/mdgpt/pretrain.yaml
```

## Notes

- Leave-one-out pretrain: `--target Cora` (excludes that dataset from source graphs).
- Checkpoints default under `ckpts/mdgpt/...`; aligners saved as `aligners.pkl` when joblib is available.
