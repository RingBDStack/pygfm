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

## COMBINE merge fixes (vs python-gfm 0.1.17)

- GRAVER finetune **dual mode**: `standard` / `cross-dataset` (external graphon + paper protocol)
- New modules: `io.py`, `graph.py`, `graphon.py` (`estimate_graphon`)
- DownPrompt CPU init, `for_finetune()`, dual trial eval, early stopping
- Compatible `.pkl` / `.pth` weights; GRAVER export format `cora.pt` loading
- Pretrain weights saved to `ckpts/<target>/` (e.g. `ckpts/cora/`)

See `Experiment-Manual.md` for Cora 1-shot reproduction.
