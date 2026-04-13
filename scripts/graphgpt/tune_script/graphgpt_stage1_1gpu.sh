#!/usr/bin/env bash
# GraphGPT Stage 1 — single-GPU smoke (default: few steps; raise STAGE1_MAX_STEPS or drop --max_steps for full training)
#
# Usage (GraphGPT / repo root):
#   chmod +x scripts/tune_script/graphgpt_stage1_1gpu.sh
#   bash scripts/tune_script/graphgpt_stage1_1gpu.sh
#
# Environment:
#   GRAPHGPT_NO_FLASH_ATTN=1  safer without FlashAttention (default on)
#   STAGE1_MAX_STEPS=500      optimizer steps (default 30 for a quick run)
#   STAGE1_GPUS=1             process count; use 1 for one GPU

set -e
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export GRAPHGPT_NO_FLASH_ATTN="${GRAPHGPT_NO_FLASH_ATTN:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM=false

NGPUS="${STAGE1_GPUS:-1}"
MAX_STEPS="${STAGE1_MAX_STEPS:-30}"

model_path="${model_path:-./ckpts/graphgpt/checkpoints/vicuna-7b-v1.5}"
instruct_ds="${instruct_ds:-./datasets/graphgpt/stage_1/graph_matching.json}"
graph_data_path="${graph_data_path:-./datasets/graphgpt/graph_data/all_graph_data.pt}"
pretra_gnn="${pretra_gnn:-clip_gt_arxiv}"
output_model="${output_model:-./ckpts/graphgpt/checkpoints/stage_1_1gpu}"

echo "ROOT=$ROOT  GPUs=$NGPUS  max_steps=$MAX_STEPS"

torchrun --nnodes=1 --nproc_per_node="${NGPUS}" --master_port="${STAGE1_MASTER_PORT:-29511}" \
  -m pygfm.baseline_models.graphgpt.train.train_mem \
  --model_name_or_path "${model_path}" \
  --version v1 \
  --data_path "${instruct_ds}" \
  --graph_content ./datasets/graphgpt/arxiv_ti_ab.json \
  --graph_data_path "${graph_data_path}" \
  --graph_tower "${pretra_gnn}" \
  --tune_graph_mlp_adapter True \
  --graph_select_layer -2 \
  --use_graph_start_end \
  --bf16 True \
  --fp16 False \
  --output_dir "${output_model}" \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 15 \
  --save_total_limit 1 \
  --learning_rate 2e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --report_to none \
  --dataloader_num_workers 0

echo "Done. Output: ${output_model}"
