# Stage 2 — single-GPU smoke / training notes

## Prerequisites

- Stage 1 finished and **`ckpts/graphgpt/checkpoints/stage_1_projector/stage_1_projector.bin`** exists
- Stage 2 instructions: `datasets/graphgpt/stage_2/arxiv_pub_node_st_cot_link_mix.json`
- Same dependency stack as [STAGE1_1GPU_README.md](./STAGE1_1GPU_README.md)

## Run

```bash
rm -rf ./ckpts/graphgpt/checkpoints/stage_2_1gpu   # recommended before retrying a failed run

export STAGE2_MAX_STEPS=30          # raise for production training
bash scripts/graphgpt/tune_script/graphgpt_stage2_1gpu.sh
```

Env overrides: `tuned_proj`, `instruct_ds`, `output_model`, `STAGE2_GPUS`.

## Code changes (same idea as Stage 1)

- `GraphLlama.py`: `weights_only=False` when loading `pretrain_graph_mlp_adapter`; load **only** `graph_projector` keys for the projector to avoid colliding with `embed_tokens`.

Default output: `./ckpts/graphgpt/checkpoints/stage_2_1gpu`.
