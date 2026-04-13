# Stage 1 — single-GPU smoke / training notes

## Dependencies (example)

```bash
pip install "fschat[model_worker]" ftfy einops sentencepiece protobuf torch_geometric
# PyTorch 2.8 + CUDA example (adjust for your stack):
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Data and graph encoder: `python scripts/graphgpt/ensure_stage1_files.py`  
Vicuna: `ckpts/graphgpt/checkpoints/vicuna-7b-v1.5/` (see `model_download_links.md`).

## Run

From **project root**:

```bash
# Remove partial outputs before retrying a failed run
rm -rf ./ckpts/graphgpt/checkpoints/stage_1_1gpu

# Default script runs ~30 steps for a quick check; increase for real training:
export STAGE1_MAX_STEPS=5000   # or drop --max_steps in the script for full epochs
bash scripts/graphgpt/tune_script/graphgpt_stage1_1gpu.sh
```

- **No FlashAttention**: scripts set `GRAPHGPT_NO_FLASH_ATTN=1`. If flash-attn is installed, `unset GRAPHGPT_NO_FLASH_ATTN` first.
- **Multi-GPU**: `export STAGE1_GPUS=4` and raise `per_device_train_batch_size` (similar to the upstream 4-GPU script).

## Compatibility tweaks in this repo (newer torch / transformers)

- `train_graph.py`: `torch.load(..., weights_only=False)`, `eval_strategy`, `Trainer(processing_class=)`, `graph_hidden_size`, etc.
- `GraphLlama.py`: infers `graph_hidden_size` from the graph encoder.
- `graphchat_trainer.py`: temporarily swaps in a valid `GenerationConfig` when saving checkpoints to avoid Vicuna `generation_config.json` clashes with newer transformers.

Default output: `./ckpts/graphgpt/checkpoints/stage_1_1gpu` (includes `graph_projector.bin`, etc.).
