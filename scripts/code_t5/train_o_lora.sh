#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0
export SCRATCH_ROOT=/data/scratch/projects/punim1928/east/CodeGR/Dense/any-ssr/.cache
mkdir -p "$SCRATCH_ROOT/torch_extensions" "$SCRATCH_ROOT/tmp"

export TORCH_EXTENSIONS_DIR=$SCRATCH_ROOT/torch_extensions
export TMPDIR=$SCRATCH_ROOT/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Salesforce/codet5p-770m \
   --data_path "" \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method O-LoRA \
   --output_dir ./output_models/t5_o_lora \
   --run_name t5_o_lora_CodeTrans \
   --group_name t5_o_lora \
   --logging_steps 100 \
   --num_eval 10 \
   --num_train 100 \
   --num_test 10 
