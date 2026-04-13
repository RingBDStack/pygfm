# Paths follow this repo’s default layout; set model_path to your local Vicuna directory.
model_path=./ckpts/graphgpt/checkpoints/vicuna-7b-v1.5
instruct_ds=./datasets/graphgpt/stage_1/graph_matching.json
graph_data_path=./datasets/graphgpt/graph_data/all_graph_data.pt
pretra_gnn=clip_gt_arxiv
output_model=./ckpts/graphgpt/checkpoints/stage_1

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
    -m pygfm.baseline_models.graphgpt.train.train_mem \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./datasets/graphgpt/arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
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
    --report_to wandb
