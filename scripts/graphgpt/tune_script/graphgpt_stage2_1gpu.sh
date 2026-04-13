#!/usr/bin/env bash
# GraphGPT Stage 2 — single-GPU smoke (requires stage_1_projector.bin)
#
# Usage (repo root):
#   bash scripts/tune_script/graphgpt_stage2_1gpu.sh
#
# Environment:
#   STAGE2_MAX_STEPS=30   default 30; raise or remove --max_steps for full training
#   STAGE2_GPUS=1
#   STAGE2_MASTER_PORT=29521

set -e
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export GRAPHGPT_NO_FLASH_ATTN="${GRAPHGPT_NO_FLASH_ATTN:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM=false

NGPUS="${STAGE2_GPUS:-1}"
MAX_STEPS="${STAGE2_MAX_STEPS:-30}"

model_path="${model_path:-./ckpts/graphgpt/checkpoints/vicuna-7b-v1.5}"
instruct_ds="${instruct_ds:-./datasets/graphgpt/stage_2/arxiv_pub_node_st_cot_link_mix.json}"
graph_data_path="${graph_data_path:-./datasets/graphgpt/graph_data/all_graph_data.pt}"
pretra_gnn="${pretra_gnn:-clip_gt_arxiv}"
tuned_proj="${tuned_proj:-./ckpts/graphgpt/checkpoints/stage_1_projector/stage_1_projector.bin}"
output_model="${output_model:-./ckpts/graphgpt/checkpoints/stage_2_1gpu}"

if [[ ! -f "$tuned_proj" ]]; then
  echo "Missing $tuned_proj — finish Stage 1 and export the projector checkpoint first."
  exit 1
fi

echo "ROOT=$ROOT  GPUs=$NGPUS  max_steps=$MAX_STEPS  tuned_proj=$tuned_proj"

torchrun --nnodes=1 --nproc_per_node="${NGPUS}" --master_port="${STAGE2_MASTER_PORT:-29521}" \
  -m pygfm.baseline_models.graphgpt.train.train_mem \
  --model_name_or_path "${model_path}" \
  --version v1 \
  --data_path "${instruct_ds}" \
  --graph_content ./datasets/graphgpt/arxiv_ti_ab.json \
  --graph_data_path "${graph_data_path}" \
  --graph_tower "${pretra_gnn}" \
  --pretrain_graph_mlp_adapter "${tuned_proj}" \
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
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --report_to none

echo "Done. Output: ${output_model}"
