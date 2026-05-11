#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0,1

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
  --data_path /path/to/LLM-CL-Benchmark_5000 \
  --dataset_name all \
  --benchmark executable \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
  --lr_scheduler_type cosine \
  --num_warmup_steps 0 \
  --seed 1234 \
  --zero_stage 2 \
  --deepspeed \
  --print_loss \
  --learning_rate 1e-4 \
  --CL_method SeqLoRA \
  --output_dir ./output_models/SeqLoRA_Qwen2.5-Coder-1.5B_executable \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --temperature 0.2 \
  --top_p 0.95 \
  --repetition_penalty 1 \
  --do_sample \
  --num_train -1 \
  --num_eval 3 \
  --num_test -1 \
  --run_name run_1 \
  --group_name SeqLoRA_Qwen2.5-Coder-1.5B_executable \
  --max_prompt_len 1024,1024,1024,1024,1024,1024,1024,1024,1024 \
  --max_ans_len 2048,2048,2048,2048,2048,2048,2048,2048,2048 \
  --num_train_epochs 3

: "${HF_MODEL_REPO_ID:=ankhanhtran02/SeqLoRA_Qwen2.5-Coder-1.5B_executable}"

python upload_output_to_hf.py \
  --output-dir "./output_models/SeqLoRA_Qwen2.5-Coder-1.5B_executable" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload SeqLoRA executable outputs"
