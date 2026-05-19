#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=1,2,3,4
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
  --offload \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --CL_method L2P \
  --output_dir ./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --run_name run_1 \
  --group_name L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable \
  --num_train -1 \
  --num_eval 3 \
  --num_test -1 \
  --max_prompt_len 1024,1024,1024,1024,1024,1024,1024,1024,1024 \
  --max_ans_len 2048,2048,2048,2048,2048,2048,2048,2048,2048 \
  --temperature 0.2 \
  --top_p 0.95 \
  --repetition_penalty 1 \
  --do_sample \
  --num_train_epochs 3

: "${HF_MODEL_REPO_ID:=ankhanhtran02/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable}"

python upload_output_to_hf.py \
  --output-dir "./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload L2P executable outputs"