#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0

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
   --CL_method O-LoRA \
   --output_dir ./output_models/OLoRA_Qwen2.5-Coder-1.5B_with_instruction_pool \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --do_sample \
   --temperature 0.2 \
   --top_p 0.95 \
   --repetition_penalty 1.2 \
   --run_name run_1 \
   --group_name OLoRA_Qwen2.5-Coder-1.5B_with_instruction_pool