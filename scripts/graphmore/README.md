# GraphMoRE

GraphMoRE baseline training and downstream scripts.

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/graphmore/pretrain.py` |
| Node finetune | `python scripts/graphmore/finetune.py` |
| Graph finetune | `python scripts/graphmore/finetune_graph.py` |
| Experiment driver | `python scripts/graphmore/run_experiments.py` |

YAML template: `configs/_templates/gfm_preprompt_pretrain.yaml` → `configs/graphmore/pretrain.yaml` (adjust `save_dir` / `data_root`).
