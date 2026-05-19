#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Salesforce/codet5p-770m \
   --data_path "" \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method EWC \
   --output_dir ./output_models/t5_ewc/CodeTrans \
   --run_name t5_ewc_CodeTrans \
   --group_name t5_ewc_CodeTrans \
   --logging_steps 10 \
   --num_train 2500 \
   --num_eval 100 \
   --num_test 100
