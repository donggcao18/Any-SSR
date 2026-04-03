#!bin/bash
BASE_PATH="/U_PZL2023ZZ0005/rhe/Any-SSR/output_models"
port=$(shuf -i25000-30000 -n1)

deepseed_cmd="deepspeed --include=localhost:2 --master_port $port"

# HF code task order (must match router training)
HF_TASKS="hf:CONCODE,hf:CodeTrans,hf:CodeSearchNet,hf:BFP,hf:TheVault_Csharp,hf:KodCode,hf:RunBugRun,hf:CoST"

$deepseed_cmd inference/infer_anyssr_total.py \
   --router_weight_path "$BASE_PATH/router_weights" \
   --data_path "" \
   --inference_tasks $HF_TASKS \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --inference_model_path "$BASE_PATH/hf_CONCODE/0","$BASE_PATH/hf_CodeTrans/0","$BASE_PATH/hf_CodeSearchNet/0","$BASE_PATH/hf_BFP/0","$BASE_PATH/hf_TheVault_Csharp/0","$BASE_PATH/hf_KodCode/0","$BASE_PATH/hf_RunBugRun/0","$BASE_PATH/hf_CoST/0" \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --inference_output_path /U_PZL2023ZZ0005/rhe/Any-SSR/inference_result

