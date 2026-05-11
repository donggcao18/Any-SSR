#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0,1

: "${DEBUG_DISTRIBUTED:=0}"
: "${LOG_DIR:=./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable/deepspeed_logs}"

if [[ "$DEBUG_DISTRIBUTED" == "1" ]]; then
  export PYTHONFAULTHANDLER=1
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_DEBUG=INFO
  export NCCL_ASYNC_ERROR_HANDLING=1
  export DEEPSPEED_LOG_LEVEL=debug
fi

set -euo pipefail

port=$(shuf -i25000-30000 -n1)
mkdir -p "$LOG_DIR"

deepspeed --master_port "$port" --log_dir "$LOG_DIR" training/main_anamoe.py \
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
  --CL_method L2P \
  --output_dir ./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --run_name run_1 \
  --group_name L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable \
  --num_train 100 \
  --num_eval 10 \
  --num_test 10 \
  --max_prompt_len 1024,1024,1024,1024,1024,1024,1024,1024,1024 \
  --max_ans_len 2048,2048,2048,2048,2048,2048,2048,2048,2048 \
  --temperature 0.2 \
  --top_p 0.95 \
  --repetition_penalty 1 \
  --do_sample \
  --num_train_epochs 2 

: "${HF_MODEL_REPO_ID:=ankhanhtran02/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable}"

python upload_output_to_hf.py \
  --output-dir "./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload L2P executable outputs"