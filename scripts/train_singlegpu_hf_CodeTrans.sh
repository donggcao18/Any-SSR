#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

python training/main_singlegpu.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --dataset_name CodeTrans \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --max_prompt_len 320 \
   --max_ans_len 256 \
   --learning_rate 2e-4 \
   --num_train_epochs 3 \
   --num_warmup_steps 0 \
   --seed 1234 \
   --print_loss \
   --logging_steps 10 \
   --CL_method anamoe \
   --lora_dim 16 \
   --lora_alpha 32 \
   --lora_dropout 0.1 \
   --output_dir ./output_models/anamoe_singlegpu/CodeTrans
