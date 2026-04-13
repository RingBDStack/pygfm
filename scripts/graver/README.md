# GRAVER

GRAVER baseline (PrePrompt / DownPrompt) scripts.

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/graver/pretrain.py` |
| Node finetune | `python scripts/graver/finetune.py` |
| Graph finetune | `python scripts/graver/finetune_graph.py` |
| 1-shot sweep | `python scripts/graver/run_1shot_100task.py` |

YAML template: `configs/_templates/gfm_preprompt_pretrain.yaml` → `configs/graver/pretrain.yaml` with paths under `ckpts/graver` / `datasets/graver`.
