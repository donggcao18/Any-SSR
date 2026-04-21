#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache


# ---------- Run ----------
python train_router_ana_continual.py \
    --model          "Qwen/Qwen2.5-Coder-1.5B" \
    --cuda_devices   "0,1" \
    --feature_layers 4 \
    --gamma          10000 \
    --router_weights_path "output_models/router_weights_with_pool" \
    --dataset_cache_path  "output_models/outputs_router_dataset_with_pool_cache" \
    --dataset_path        "dataset/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
    --max_prompt_len 512 \
    --max_ans_len    256 \
    --batch_size     1 \
    --rls_lambda     100.0 \
    --tasks hf:CONCODE hf:CodeTrans hf:CodeSearchNet hf:BFP hf:KodCode hf:TheVault_Csharp hf:RunBugRun hf:CoST


python eval_router_ana.py \
    --model               "Qwen/Qwen2.5-Coder-1.5B" \
    --cuda_devices        "0" \
    --feature_layers      4 \
    --gamma               10000 \
    --router_weights_path "output_models/router_weights_with_pool" \
    --dataset_cache_path  "output_models/outputs_router_dataset_with_pool_cache" \
    --dataset_path        "dataset/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
    --max_prompt_len      512 \
    --max_ans_len         256 \
    --batch_size          1 \
    --log_file            "output_models/logs/eval_router_with_pool.log" \
    --tasks hf:CONCODE hf:CodeTrans hf:CodeSearchNet hf:BFP hf:KodCode hf:TheVault_Csharp hf:RunBugRun hf:CoST
