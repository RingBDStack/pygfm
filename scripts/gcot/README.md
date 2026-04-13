# GCoT

Graph Chain-of-Thought baseline in this repo (PrePrompt / DownPrompt stack aligned with other GFMs).

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/gcot/pretrain.py` |
| Splits | `python scripts/gcot/generate_downstream.py` |
| Node finetune | `python scripts/gcot/finetune.py` |
| Graph finetune | `python scripts/gcot/finetune_graph.py` |
| 1-shot sweep | `python scripts/gcot/run_1shot_100task.py` |

YAML template: `configs/_templates/gfm_preprompt_pretrain.yaml` → copy to `configs/gcot/pretrain.yaml` with `ckpts/gcot` / `datasets/gcot`.
