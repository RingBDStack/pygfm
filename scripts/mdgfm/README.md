# MDGFM

MDGFM extends the MDGPT-style pipeline with **per-domain pretext + shared sumtext** before the GCN and contrastive loss.

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments (repo root)

| Step | Command |
|------|---------|
| Pretrain | `python scripts/mdgfm/pretrain.py -c configs/_templates/gfm_preprompt_pretrain.yaml` (copy & set `save_dir` to `ckpts/mdgfm`, `data_root` to `datasets/mdgfm`) |
| Few-shot splits | `python scripts/mdgfm/generate_downstream.py few_shot --dataset Cora --k_shot 5` |
| Node finetune | `python scripts/mdgfm/finetune.py --ckpt ...` |
| Graph finetune | `python scripts/mdgfm/finetune_graph.py --ckpt ...` |
| 1-shot batch | `python scripts/mdgfm/run_1shot_100task.py` |

YAML: supports `-c` / `--config` on pretrain (see `parse_args_with_optional_yaml` in `pretrain.py`).

Template: copy `configs/_templates/gfm_preprompt_pretrain.yaml` → `configs/mdgfm/pretrain.yaml` and fix paths.
