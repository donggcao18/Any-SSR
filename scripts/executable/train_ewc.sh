#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0
# This script uses 1 GPU. Use a larger disk space (56GB) to save the model checkpoints (full model).

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
  --data_path /path/to/LLM-CL-Benchmark_5000 \
  --dataset_name all \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
  --benchmark executable \
  --lr_scheduler_type cosine \
  --num_warmup_steps 0 \
  --seed 1234 \
  --zero_stage 2 \
  --deepspeed \
  --print_loss \
  --learning_rate 1e-4 \
  --CL_method EWC \
  --output_dir ./output_models/EWC_Qwen2.5-Coder-1.5B_with_instruction_pool_executable \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --max_prompt_len 1024 \
  --max_ans_len 2048 \
  --num_train_epochs 2 \
  --run_name "run_1" \
  --group_name "EWC_Qwen2.5-Coder-1.5B_with_instruction_pool_executable" \
  --num_train 100 \
  --num_eval 3 \
  --num_test 5 \
  --logging_steps 10 \
  --temperature 0.2 \
  --top_p 0.95 \
  --repetition_penalty 1 \
  --do_sample 

: "${HF_MODEL_REPO_ID:=ankhanhtran02/lora-per-task-executable-start-0}"

python upload_output_to_hf.py \
  --output-dir "./output_models/EWC_Qwen2.5-Coder-1.5B_with_instruction_pool_executable" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload EWC executable outputs"