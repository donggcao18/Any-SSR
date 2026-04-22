#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=1

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name CodeTrans \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 320 \
   --max_ans_len 256 \
   --learning_rate 1e-4 \
   --num_train_epochs 3 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method anamoe \
   --output_dir ./output_models/anamoe/CodeTrans \
   # --weight_decay 0. \

