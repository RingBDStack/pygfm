# BRIDGE

Same directory conventions as MDGPT (`datasets/bridge`, `ckpts/bridge`, `downstream_data/bridge`).

## Install

```bash
cd /path/to/repo
pip install -e .
```

Run `python scripts/bridge/*.py` from the repository root.

## Model (`pygfm.baseline_models.bridge`)

| Module | Role |
|--------|------|
| `BridgePrePromptModel` | Domain mask + 3-layer GCN + contrastive + variance regularizer |
| `BridgeDownPromptModel` | Frozen backbone + MoE mask + prototypes + spectral reg + routing entropy (node) |
| `BridgeDownPromptGraphModel` | Above + subgraph mean-pool / disjoint batch graph forward |

## Data layout (`bridge` name tag)

- Raw graphs: `datasets/bridge` (can symlink to `datasets/mdgpt`)
- Pretrain ckpt: `ckpts/bridge/{dataset}/preprompt_{dataset}.pth`
- Downstream splits: `downstream_data/bridge/{dataset}/{1|5}shot/splits.pt` and `{k}shot_graph_batch/splits.pt`

## Commands

```bash
# 1) Leave-one-out pretrain (exclude Cora)
python scripts/bridge/pretrain.py --target Cora --row_norm

# 2) Few-shot splits
python scripts/bridge/generate_downstream.py few_shot --dataset Cora --k_shot 1 --data_root datasets/bridge

# 3) Node classification finetune
python scripts/bridge/finetune.py --dataset Cora --k_shot 1 \
  --ckpt ckpts/bridge/cora/preprompt_cora.pth --row_norm

# 4) Graph-level few-shot (generate graph_batch first)
python scripts/bridge/generate_downstream.py graph_batch --dataset Cora --k_shot 1 --data_root datasets/bridge
python scripts/bridge/finetune_graph.py --dataset Cora --k_shot 1 \
  --ckpt ckpts/bridge/cora/preprompt_cora.pth

# 5) Batch 1-shot × 100 tasks
python scripts/bridge/run_1shot_100task.py
```

**Note**: `--row_norm` matches common BRIDGE setups (row-normalize then PCA). Drop it to align with MDGPT-style PCA-only.

YAML template: copy `configs/_templates/gfm_preprompt_pretrain.yaml` to `configs/bridge/pretrain.yaml` with `save_dir: ckpts/bridge` and `data_root: datasets/bridge`.
