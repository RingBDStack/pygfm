# HGPrompt

Heterogeneous-graph prompt baseline (PrePrompt / DownPrompt integrated under `pygfm.baseline_models.hgprompt`).

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/hgprompt/pretrain.py` |
| Splits | `python scripts/hgprompt/generate_downstream.py` |
| Node finetune | `python scripts/hgprompt/finetune.py` |
| 1-shot sweep | `python scripts/hgprompt/run_1shot_100task.py` |

YAML template: copy `configs/_templates/gfm_preprompt_pretrain.yaml` to `configs/hgprompt/pretrain.yaml` and set `save_dir` / `data_root` for HGPrompt data.
