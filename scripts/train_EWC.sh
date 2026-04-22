#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=2,3

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name all \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --learning_rate 1e-4 \
   --CL_method EWC \
   --output_dir ./output_models/EWC_Qwen2.5-Coder-1.5B_with_instruction_pool \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 2 \