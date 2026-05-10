#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

# ---------- Run (Executable benchmark) ----------
python train_router_ana_continual.py \
    --model_name_or_path "Qwen/Qwen2.5-Coder-1.5B" \
    --benchmark executable \
    --router_weights_path "output_models/router_weights_executable" \
    --dataset_cache_path  "output_models/outputs_router_executable_cache" \
    --max_prompt_len 512 \
    --max_ans_len    256 \
    --batch_size     1

python eval_router_ana.py \
    --model_name_or_path "Qwen/Qwen2.5-Coder-1.5B" \
    --benchmark executable \
    --router_weights_path "output_models/router_weights_executable" \
    --dataset_cache_path  "output_models/outputs_router_executable_cache" \
    --max_prompt_len      512 \
    --max_ans_len         256 \
    --batch_size          1
