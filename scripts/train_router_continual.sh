#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache


# ---------- Run ----------
python train_router_ana_continual.py \
    --model          "codellama/CodeLlama-7b-hf" \
    --cuda_devices   "0,1" \
    --feature_layers 4 \
    --gamma          10000 \
    --router_weights_path "output_models/router_weights" \
    --dataset_cache_path  "output_models/outputs_router_dataset_cache" \
    --dataset_path        "dataset/TRACE-Benchmark/LLM-CL-Benchmark_5000" \
    --max_prompt_len 512 \
    --max_ans_len    256 \
    --batch_size     128 \
    --rls_lambda     100.0 \
    --tasks hf:CONCODE hf:CodeTrans hf:CodeSearchNet hf:BFP hf:TheVault_Csharp hf:KodCode hf:RunBugRun hf:CoST
