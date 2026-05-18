#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

checkpoint_root=./output_models/OLoRA_Qwen2.5-Coder-1.5B
output_root=${checkpoint_root}/predictions
model_name_or_path=Qwen/Qwen2.5-Coder-1.5B

start_task_id=0
end_task_id=7

per_device_eval_batch_size=32
lora_alpha=32
lora_r=16
num_test=-1
seed=1234

mkdir -p "$output_root"

python inference/infer_olora_final.py \
   --checkpoint_path "$checkpoint_root" \
   --start_task_id "$start_task_id" \
   --end_task_id "$end_task_id" \
   --model_name_or_path "$model_name_or_path" \
   --per_device_eval_batch_size "$per_device_eval_batch_size" \
   --lora_alpha "$lora_alpha" \
   --lora_r "$lora_r" \
   --num_test "$num_test" \
   --seed "$seed" \
   --output_dir "$output_root"
