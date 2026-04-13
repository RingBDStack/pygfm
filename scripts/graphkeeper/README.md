# GraphKeeper

GraphKeeper-style **chain prediction pretraining** and **LoRA**, wired into the same script/data layout as MDGPT (PyG; no DGL domain-incremental expert loop in this port).

## Install

```bash
cd /path/to/repo
pip install -e .
```

Run `python scripts/graphkeeper/*.py` from the repository root.

## Differences from the original (read this)

| Original `new_model/GraphKeeper` | This port |
|----------------------------------|-----------|
| DGL, `NET.observe` multi-task incremental | Multi-domain single-graph PrePrompt (same as MDGPT) |
| `GCN_LoRA` per-layer LoRA | `EmbeddingLoRA` on node embeddings (low-rank residual) |
| `Regression` + analytic linear layer | `TaskHead` prototype matching (same as MDGPT DownPrompt) |

Full **Domain-IL / multi-expert / DBSCAN prototypes** still belong to the official repo; this port targets **unified GFM pretraining + few-shot downstream**.

## Models

- `GraphKeeperPrePromptModel`: `NodeLevelPrompt` per domain + shared GCN + **chain-prediction BCE** (+ optional contrastive loss).
- `GraphKeeperDownPromptModel` / `GraphKeeperDownPromptGraphModel`: frozen GCN, train **prompt + LoRA**.

## Paths

- Data: `datasets/graphkeeper` (can be the same tree as `datasets/mdgpt`)
- Checkpoints: `ckpts/graphkeeper/{datasets}/preprompt.pth`, or leave-one-domain `preprompt_{target}.pth` (`--datasets` names the output subdir; multi-domain list uses `--domains`)
- Downstream: `downstream_data/graphkeeper/...`

## Example commands

```bash
# Pretrain (leave-one-domain Cora, save under ckpts/graphkeeper/cora/, or pass --datasets my_run)
python scripts/graphkeeper/pretrain.py --target Cora --no_swanlab

# Multi-domain sequential incremental (example)
python scripts/graphkeeper/pretrain.py --domains Cora,Citeseer,Pubmed,Photo,Computers --datasets incremental_run --no_swanlab

# Add MDGPT-style contrastive (weight 0.5)
python scripts/graphkeeper/pretrain.py --target Cora --contrastive_weight 0.5 --no_swanlab

python scripts/graphkeeper/generate_downstream.py few_shot --dataset Cora --k_shot 1 --data_root datasets/graphkeeper

python scripts/graphkeeper/finetune.py --dataset Cora --k_shot 1 \
  --ckpt ckpts/graphkeeper/cora/preprompt_cora.pth --no_swanlab
# If you pretrained with --datasets my_run, use --ckpt ckpts/graphkeeper/my_run/preprompt.pth
```

## Citation (original paper)

See BibTeX in `new_model/GraphKeeper/README.md`.
