# SAMGPT

SAMGPT-style **multi-source prompts + shared GCN + contrastive** pretrain, with downstream DownPrompt variants.

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/samgpt/pretrain.py` (`-c` YAML supported) |
| Generate splits | `python scripts/samgpt/generate_downstream.py` |
| Node finetune | `python scripts/samgpt/finetune.py` |
| Graph finetune | `python scripts/samgpt/finetune_graph.py` |
| 1-shot sweep | `python scripts/samgpt/run_1shot_100task.py` |

YAML template: start from `configs/_templates/gfm_preprompt_pretrain.yaml`, set `save_dir: ckpts/samgpt` and `data_root: datasets/samgpt` (or your layout).
