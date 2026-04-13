# GraphPrompt

GraphPrompt-style pretrain + DownPrompt downstream (see `pygfm.baseline_models.graphprompt`).

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/graphprompt/pretrain.py` |
| Splits | `python scripts/graphprompt/generate_downstream.py` |
| Node finetune | `python scripts/graphprompt/finetune.py` |
| Graph finetune | `python scripts/graphprompt/finetune_graph.py` |
| 1-shot sweep | `python scripts/graphprompt/run_1shot_100task.py` |

YAML template: `configs/_templates/gfm_preprompt_pretrain.yaml` → `configs/graphprompt/pretrain.yaml`.
