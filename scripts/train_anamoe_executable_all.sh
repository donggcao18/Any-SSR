#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0,1

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

# Executable benchmark tasks: python, cpp, swift, rust, csharp, java, php, typescript, shell
# Using max_prompt_len=1024 and max_new_tokens=2048 for all tasks.

for dataset in python cpp swift rust csharp java php typescript shell; do
  deepspeed --master_port "$port" training/main_anamoe.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
    --benchmark executable \
    --data_path "" \
    --dataset_name "$dataset" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_prompt_len 1024 \
    --max_ans_len 2048 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method anamoe \
    --repetition_penalty 1 \
    --do_sample \
    --disable_epoch_eval \
    --output_dir "./output_models/anamoe_executable/${dataset}" \
    --run_name "anamoe_${dataset}" \
    --group_name "anamoe_executable_all" \
    --num_train 100 \
    --num_eval 50 \
    --num_test 50 \
    --logging_steps 10

done
