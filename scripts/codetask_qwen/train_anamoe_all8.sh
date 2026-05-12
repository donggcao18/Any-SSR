#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0,1

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

# Task order: CONCODE, CodeTrans, CodeSearchNet, BFP, KodCode, RunBugRun, TheVault_Csharp, CoST
# max_prompt_len: 320,   320,       256,          130, 512,     256,       256,              256
# max_ans_len:    150,   256,       128,           120, 300,     128,       128,              128


# ─────────────────────────────────────────────
# 2. CodeTrans  (max_prompt=320  max_ans=256)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name CodeTrans \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 320 \
   --max_ans_len 256 \
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
   --output_dir ./output_models/anamoe/CodeTrans \
   --run_name anamoe_CodeTrans \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10

# ─────────────────────────────────────────────
# 3. CodeSearchNet  (max_prompt=256  max_ans=128)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name CodeSearchNet \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 256 \
   --max_ans_len 128 \
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
   --output_dir ./output_models/anamoe/CodeSearchNet \
   --run_name anamoe_CodeSearchNet \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 

# ─────────────────────────────────────────────
# 1. CONCODE  (max_prompt=320  max_ans=150)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name CONCODE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 320 \
   --max_ans_len 150 \
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
   --output_dir ./output_models/anamoe/CONCODE \
   --run_name anamoe_CONCODE \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 



# ─────────────────────────────────────────────
# 4. BFP  (max_prompt=130  max_ans=120)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name BFP \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 130 \
   --max_ans_len 120 \
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
   --output_dir ./output_models/anamoe/BFP \
   --run_name anamoe_BFP \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 



# ─────────────────────────────────────────────
# 6. RunBugRun  (max_prompt=256  max_ans=128)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name RunBugRun \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 256 \
   --max_ans_len 128 \
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
   --output_dir ./output_models/anamoe/RunBugRun \
   --run_name anamoe_RunBugRun \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 

# ─────────────────────────────────────────────
# 7. TheVault_Csharp  (max_prompt=256  max_ans=128)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name TheVault_Csharp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 256 \
   --max_ans_len 128 \
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
   --output_dir ./output_models/anamoe/TheVault_Csharp \
   --run_name anamoe_TheVault_Csharp \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 

# ─────────────────────────────────────────────
# 8. CoST  (max_prompt=256  max_ans=128)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name CoST \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 256 \
   --max_ans_len 128 \
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
   --output_dir ./output_models/anamoe/CoST \
   --run_name anamoe_CoST \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 


# ─────────────────────────────────────────────
# 5. KodCode  (max_prompt=512  max_ans=300)
# ─────────────────────────────────────────────
deepspeed --master_port "$port" training/main_anamoe.py \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --data_path "" \
   --dataset_name KodCode \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --max_prompt_len 512 \
   --max_ans_len 300 \
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
   --output_dir ./output_models/anamoe/KodCode \
   --run_name anamoe_KodCode \
   --group_name anamoe_all8 \
   --num_eval 100 \
   --logging_steps 10 