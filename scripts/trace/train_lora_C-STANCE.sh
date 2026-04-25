#!bin/bash
port=$(shuf -i25000-30000 -n1)
echo "使用端口: $port"

# port=28888
# export CUDA_VISIBLE_DEVICES=0,1
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=$port
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=INFO

deepspeed --include=localhost:2,3 --master_port $port training/main_anamoe.py \
   --data_path /U_PZL2023ZZ0005/rhe/dataset/TRACE-Benchmark/LLM-CL-Benchmark_5000/ \
   --dataset_name C-STANCE \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 10 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method anamoe \
   --output_dir /U_PZL2023ZZ0005/rhe/Any-SSR/output_models/C-STANCE
