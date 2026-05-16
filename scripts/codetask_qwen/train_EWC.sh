#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=1
export SCRATCH_ROOT=/data/scratch/projects/punim1928/east/CodeGR/Dense/any-ssr/.cache
mkdir -p "$SCRATCH_ROOT/torch_extensions" "$SCRATCH_ROOT/tmp"

export TORCH_EXTENSIONS_DIR=$SCRATCH_ROOT/torch_extensions
export TMPDIR=$SCRATCH_ROOT/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
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
   --output_dir ./output_models/EWC_Qwen2.5-Coder-1.5B \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 4 \
   --run_name run_1 \
   --group_name EWC_Qwen2.5-Coder-1.5B \
   --num_train -1 \
   --num_eval 10 \
   --num_test -1 