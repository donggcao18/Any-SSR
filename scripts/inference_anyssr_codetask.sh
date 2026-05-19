#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/U_PZL2023ZZ0005/rhe/Any-SSR/output_models"
port=$(shuf -i25000-30000 -n1)
GPU_ID="${GPU_ID:-0}"
INFERENCE_BATCH="${INFERENCE_BATCH:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-inference_result_anyssr_codetask}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DS_ACCELERATOR=cuda

deepseed_cmd="deepspeed --include=localhost:${GPU_ID} --master_port $port"

# HF code task order (must match router training)
HF_TASKS="CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST"

mkdir -p logs

$deepseed_cmd inference/infer_anyssr_total.py \
   --router_weight_path "ankhanhtran02/router_weights_codetask_qwen25_coder_15b" \
   --benchmark non-executable \
   --data_path "" \
   --inference_tasks $HF_TASKS \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --base_path dongg18/anamoe \
   --inference_model_path "CONCODE/0","CodeTrans/0","CodeSearchNet/0","BFP/0","KodCode/0","RunBugRun/0","TheVault_Csharp/0","CoST/0" \
   --seed 1234 \
   --deepspeed \
   --device cuda \
   --inference_output_path "$OUTPUT_DIR" \
   --inference_batch "$INFERENCE_BATCH" \
   2>&1 | tee logs/train.log
