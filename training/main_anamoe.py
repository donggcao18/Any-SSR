#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9576))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

from asyncio import run
import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
import datetime
import wandb
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class TeeLogger:
    """Duplicates stdout writes to both the terminal and a log file."""
    def __init__(self, filepath):
        self._terminal = sys.__stdout__
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._log = open(filepath, 'a', buffering=1)  # line-buffered
        self._log.write(f"\n{'='*60}\n")
        self._log.write(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log.write(f"{'='*60}\n")
        self._log.flush()

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    def __del__(self):
        try:
            self._log.close()
        except Exception:
            pass

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, create_codetask_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# add flash attention
try:
    from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
    from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()
    replace_bloom_attn_with_flash_attn()
except Exception:
    print("[INFO] flash-attn is unavailable; fallback to standard attention.")

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model


from params import Method2Class, AllDatasetName

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9576))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# TODO, check support for OPT and llama

def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='Qwen/Qwen2.5-Coder-1.5B',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_train",
        type=list_of_strings,
        default='-1',
        help="Number of training examples for each dataset, -1 means using all the data.",
    )
    parser.add_argument(
        "--num_eval",
        type=list_of_strings,
        default='-1',
        help="Number of evaluation examples for each dataset, -1 means using all the data.",
    )
    parser.add_argument(
        "--num_test",
        type=list_of_strings,
        default='-1',
        help="Number of test examples for each dataset, -1 means using all the data.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=list_of_strings,
        default='320,320,256,130,512,256,256,256',
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=list_of_strings,
        default='150,256,128,120,300,128,128,128',
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default='3,3,3,3,3,3,3,3',
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # LoRA related arguments
    parser.add_argument('--lora_dim',
                        type=int,
                        default=16,
                        help='LoRA dimension')
    parser.add_argument('--lora_alpha',
                        type=int,
                        default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout',
                        type=float,
                        default=0.1,
                        help='LoRA dropout')
    # parser.add_argument('--lora_target_modules',
    #                     type=list_of_strings,
    #                     default="q_proj,v_proj",
    #                     help='LoRA target modules')
    # added by wangxiao
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')

    # Generation configs
    parser.add_argument('--do_sample',
                        action='store_true',
                        help='Whether to use sampling for generation.')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.2,
                        help='Temperature for generation.')
    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='Top-p for generation.')
    parser.add_argument('--repetition_penalty',
                        type=float,
                        default=1.2,
                        help='Repetition penalty for generation.')
    parser.add_argument('--run_name',
                    type=str,
                    required=True,
                    help='Name of the run for wandb logging.')
    parser.add_argument('--group_name',
                type=str,
                default=True,
                help='Group name for wandb logging.')
    parser.add_argument('--enable_wandb',
                action='store_true',
                help='Enable wandb logging.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    class ArgsDict(dict):
        """Dictionary-style args with backward-compatible attribute access."""
        __getattr__ = dict.get

        def __setattr__(self, key, value):
            self[key] = value

    args_dict = ArgsDict(vars(args))


    if args.enable_wandb:
        run = wandb.init(
            project="CL4Code",
            group=args.group_name,
            job_type="train",
            name=f"{args.group_name}_{args.run_name}",
            config=dict(args_dict)
        )

        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")

        run.define_metric("epoch")
        run.define_metric("eval_epoch/*", step_metric="epoch")

        run.define_metric("task_id")
        run.define_metric("eval_task/*", step_metric="task_id")

    if args.local_rank == -1:
        device = torch.device("cuda")
        # torch.distributed.init_process_group(backend='nccl')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        # print("Before initialization")
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        
        # print("环境变量检查:")
        # for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        #     print(f"  {var}: {os.environ.get(var)}")

        deepspeed.init_distributed()
        # deepspeed.init_distributed(dist_backend='nccl')
        # print("After initialization")
    args.global_rank = torch.distributed.get_rank()

    # Set up file logging (rank 0 only)
    if args.global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "training.log")
        sys.stdout = TeeLogger(log_path)
        print(f"Logging to {log_path}")
        print(f"Args: {args}")

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")
    # set batch size
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps



    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout
                            )
    
    # some CL methods can be realized by peft
    if args.CL_method == "LFPT5":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        initial_prompt = getInitialPrompt(tokenizer, prompt_token_number=300)
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=300,
            prompt_tuning_init_text=initial_prompt,
            tokenizer_name_or_path=args.model_name_or_path,
        )
        model = get_peft_model(model, peft_config)

    if args.CL_method == "O-LoRA":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_dim, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False
                
    if args.CL_method == "OGD":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True

    if args.CL_method == "lora":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True

    if args.CL_method == "anamoe":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )

        names = [name for name, param in model.named_parameters()]
        # print(names)

        start = 4
        end = 32
        # filtered_names = [name for name in names if start <= int(name.split('.')[2]) < end]

        filtered_names = [
            name[:-7] for name in names
            if name.startswith("model.layers.")  # 确保是层相关的内容
            and start <= int(name.split('.')[2]) < end  # 检查层号是否在范围内
        ]

        filtered_names = [name for name in filtered_names if 'layernorm' not in name]
        filtered_names = [name for name in filtered_names if 'k_proj' not in name]
        filtered_names = [name for name in filtered_names if 'o_proj' not in name]
        filtered_names = [name for name in filtered_names if 'mlp' not in name]
        peft_config.target_modules = filtered_names

        model = get_peft_model(model, peft_config)
        # for name, param in model.named_parameters():
        #     if name.find("lora") != -1:
        #         param.requires_grad = True

        # import re
        # peft_pattern = r"^base_model.model.model.layers.(4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31).*$"
        # for name, param in model.named_parameters():
        #     if not re.match(peft_pattern, name) and name not in ['base_model.model.model.norm.weight', 'base_model.model.lm_head.weight']:
        #         param.requires_grad = False
    
    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}


    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    
    if len(args.num_train) == 1:
        args.num_train = [args.num_train[0]] * len(Datasets)
    if len(args.num_eval) == 1:
        args.num_eval = [args.num_eval[0]] * len(Datasets)
    if len(args.num_test) == 1:
        args.num_test = [args.num_test[0]] * len(Datasets)
    assert len(args.num_train) == len(Datasets), "The number of training examples should be specified for each dataset"
    assert len(args.num_eval) == len(Datasets), "The number of evaluation examples should be specified for each dataset"
    assert len(args.num_test) == len(Datasets), "The number of test examples should be specified for each dataset"
    assert len(args.max_prompt_len) == len(Datasets), "The max prompt length should be specified for each dataset"
    assert len(args.max_ans_len) == len(Datasets), "The max answer length should be specified for each dataset"
    
    for i, dataset in enumerate(Datasets):
        # dataset_path = os.path.join(args.data_path,dataset)
        # Prepare the data
        train_dataset, eval_dataset, test_dataset = create_codetask_dataset(dataset, args.seed, args.num_train[i], args.num_eval[i], args.num_test[i])

        # DataLoaders creation:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            test_sampler = SequentialSampler(test_dataset)

        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
            test_sampler = DistributedSampler(test_dataset)

        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=int(args.max_prompt_len[i]),
            max_ans_len=int(args.max_ans_len[i]),
            pad_to_multiple_of=8,
            inference=False
        )
        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=int(args.max_prompt_len[i]),
            max_ans_len=int(args.max_ans_len[i]),
            pad_to_multiple_of=8,
            inference=True
        )
                

        train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=data_collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset,
                            collate_fn=inf_data_collator,
                            sampler=test_sampler,
                            batch_size=args.per_device_eval_batch_size)
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader


    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                # TODO, check output
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def get_optimizer(model):
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
        
        total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
        num_update_steps_per_epoch = math.ceil(
            total_train_dataloader_len / args.gradient_accumulation_steps)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        
        return optimizer, lr_scheduler
    
    if args.CL_method=="PP" or args.CL_method=="L2P":
        embed_tokens = None
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            # llama/qwen and many decoder-only HF models
            embed_tokens = model.model.embed_tokens
        elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "embed_tokens"):
            # opt-like models
            embed_tokens = model.model.decoder.embed_tokens

        if embed_tokens is None:
            raise ValueError(
                f"Unsupported model architecture for {args.CL_method}: cannot locate embed_tokens on {type(model)}"
            )

        embed_tokens_shape = embed_tokens.weight.shape
        args.embed_tokens_dim = embed_tokens_shape[1]
        args.embed_tokens_length = embed_tokens_shape[0]
        args.embed_tokens = embed_tokens
            
        if args.CL_method=="PP":
            args.prefix_len = 20
            args.task_length = len(train_task_list)
            model = convert_PP_model(model, args)
            # Restrict trainable parameters for PP before DeepSpeed/optimizer init.
            # This avoids ZeRO-2 reduction over large frozen backbone params.
            for _, params in model.named_parameters():
                params.requires_grad = False
            model.model.prompt.requires_grad = True
            for params in model.model.mlps.parameters():
                params.requires_grad = True
            
        elif args.CL_method=="L2P":
            args.pool_size = 10
            args.prompt_length = 5
            args.prompt_init = "uniform"
            model = convert_L2P_model(model, args)
            for name, params in model.named_parameters():
                if "prompt" not in name:
                    params.requires_grad=False
    # print("Before initializing DeepSpeed")
    optimizer, lr_scheduler = get_optimizer(model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    # print("After initializing DeepSpeed")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****")
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # Initialize the global progress bar

    if args.CL_method in Method2Class.keys():
        CL_Trainer = Method2Class[args.CL_method](model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        CL_Trainer.train_continual()


if __name__ == "__main__":
    main()