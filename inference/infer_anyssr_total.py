"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""


# import sys 
# if hasattr(sys, 'gettrace') and sys.gettrace(): 
#     print("⚠️ 检测到已有调试器附加，强制退出！")
#     sys.exit(1) 

# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download

print('-----------------------------------------------------------------------')

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed
import json

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_codetask_dataset, create_executable_dataset, create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model

# New evaluation: BLEU + SmoothBLEU for HF tasks
from utils.code_metrics import bleu as corpus_bleu, smooth_bleu as corpus_smooth_bleu

from training.params import Method2Class, AllDatasetName,AllDatasetNameExecutable

from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model

from moe import NewSdpaAttention, NewLlamaForCausalLM, NewLlamaDecoderLayer, NewLlamaModel, NewQwen2SdpaAttention, NewQwen2ForCausalLM, NewQwen2DecoderLayer, NewQwen2Model
from transformers import GenerationConfig
from transformers.models.llama import modeling_llama, LlamaConfig
from transformers.models.qwen2 import modeling_qwen2

from lora_callback import global_callback
from peft import peft_model
import types
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG

def copy_module(module):
    new_module = types.ModuleType(module.__name__ + '_original')
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr_value = getattr(module, attr_name)
            setattr(new_module, attr_name, attr_value)
    return new_module

original_modeling_llama = copy_module(modeling_llama)
modeling_llama.LlamaModel = NewLlamaModel
modeling_llama.LlamaForCausalLM = NewLlamaForCausalLM
modeling_llama.LlamaDecoderLayer = NewLlamaDecoderLayer
modeling_llama.LlamaSdpaAttention = NewSdpaAttention

original_modeling_qwen2 = copy_module(modeling_qwen2)
modeling_qwen2.Qwen2Model = NewQwen2Model
modeling_qwen2.Qwen2ForCausalLM = NewQwen2ForCausalLM
modeling_qwen2.Qwen2DecoderLayer = NewQwen2DecoderLayer
modeling_qwen2.Qwen2SdpaAttention = NewQwen2SdpaAttention

def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument('--router_weight_path',
                        type=str,
                        default='',
                        help='Path to the router weights. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument('--benchmark',
                    type=str,
                    choices=['executable', 'non-executable'],
                    default='non-executable',
                    help='Benchmark to be evaluated: executable or non-executable')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='Qwen/Qwen2.5-Coder-1.5B',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )    
    parser.add_argument(
        "--base_path",
        type=str,
        default='dongg18/anamoe',
        help=
        "Path to trained adapter model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        nargs='+',
        required=True,
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
        "--inference_batch",
        type=int,
        default=1,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
            default=None,
            help='continual learning method used')
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
    parser.add_argument('--top_k',
                        type=int,
                        default=-1,
                        help='Top-k for generation (0 disables top-k sampling).')
    parser.add_argument('--repetition_penalty',
                        type=float,
                        default=1.0,
                        help='Repetition penalty for generation.')
    parser.add_argument('--num_return_sequences',
                        type=int,
                        default=5,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--device',
                        type=str,
                        default='auto',
                        help="Device to run on: auto, cpu, cuda, or cuda:<index>.")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def resolve_device(args) -> torch.device:
    if args.device != "auto":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot see any CUDA device. "
                "Check that the remote environment has an NVIDIA driver, a CUDA-enabled PyTorch build, "
                "and that CUDA_VISIBLE_DEVICES is set correctly."
            )
        return torch.device(args.device)
    if torch.cuda.is_available():
        if args.local_rank is not None and args.local_rank >= 0:
            return torch.device(f"cuda:{args.local_rank}")
        return torch.device("cuda")
    raise RuntimeError(
        "No CUDA device is visible. This inference script is configured to use GPU; "
        "run with --device cpu only if you intentionally want slow CPU inference."
    )


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args)
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
        print(f"[INFO] CUDA device name: {torch.cuda.get_device_name(device)}")
    if args.inference_tasks[0] == "all":
        if args.benchmark == "non-executable":    
            inference_tasks = AllDatasetName
        else:
            inference_tasks = AllDatasetNameExecutable
    else:
        inference_tasks = args.inference_tasks 
    task_num = len(inference_tasks)
    inference_model_path = args.inference_model_path
    inference_model_path = inference_model_path[0].split(',')
    generation_config = GenerationConfig(
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            repetition_penalty=args.repetition_penalty,
    )

    def prediction(model, tokenizer, task, test_dataloader, device, generation_config, max_ans_len=None):
        model.eval()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []

        if max_ans_len is None:
            max_ans_len = getattr(args, "max_ans_len", 256)

        is_executable = getattr(args, "benchmark", "non-executable") != "non-executable"
        if is_executable:
            num_return_sequences = int(getattr(args, "num_return_sequences", 5))
            top_k = int(getattr(args, "top_k", 0))
            generation_kwargs = generation_config.to_dict()
            generation_kwargs.update({
                "num_return_sequences": num_return_sequences,
                "top_k": top_k,
            })
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            num_return_sequences = 1
            generation_config = generation_config

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=False)
        for step, batch in enumerate(test_dataloader):
            sources_sequences += batch['sources']
            # print(f"Sources: {batch['sources']}")
            if 'gts' in batch:
                ground_truths += batch['gts']
                # print(f"Ground truths: {batch['gts']}")
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    gt = tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    ground_truths.append(gt)
                    # print(f"Ground truth decoded: {gt}")
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id

                generate_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_ans_len,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    use_cache=True,
                )

            sequences = tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            # print(f"Predicted sequences: {sequences}")
            if is_executable and num_return_sequences > 1:
                batch_preds = [
                    sequences[i:i + num_return_sequences]
                    for i in range(0, len(sequences), num_return_sequences)
                ]
                predicted_sequences.extend(batch_preds)
            else:
                predicted_sequences += sequences

            progress_bar.update(1)
            description = f"Test step {step}"
            progress_bar.set_description(description, refresh=False)

        return sources_sequences, predicted_sequences, ground_truths

    def _task_eval_from_predictions(task, sources_sequences, predicted_sequences, ground_truths):
        if task in ['CodeSearchNet', 'TheVault_Csharp']:
            calc_codebleu = False
        else:
            calc_codebleu = True
        return compute_metrics(predicted_sequences, ground_truths, calc_codebleu=calc_codebleu, language=DATASET_TO_OUTPUT_LANG.get(task, None))
    
    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                ground_truths: list, i_task: int, task: str):
        # save as a json file
        df = {"eval": evaluation_result}
        os.makedirs(args.inference_output_path, exist_ok=True)
        prediction_rows = [
            {
                "source": source,
                "ground-truth": gt,
                "prediction": pred,
            }
            for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
        ]
        df["predictions"] = prediction_rows
        output_file = os.path.join(args.inference_output_path, f"results-{i_task}-{task}.json")
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)
            file.write("\n")
        print(f"[INFO] Saved inference results to {output_file}", flush=True)

    # Only infer the last step
    for i in range(len(inference_tasks)-1, len(inference_tasks)):
        # if i == 0:
        #     continue

        tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
        model_dtype = torch.float16 if device.type == "cuda" else torch.float32
        if "llama" in args.model_name_or_path.lower():
            model = modeling_llama.LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                tasks=i+1,
                torch_dtype=model_dtype,
            )
        elif "qwen" in args.model_name_or_path.lower():
            model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                tasks=i+1,
                torch_dtype=model_dtype,
            )

        
        fe_path = hf_hub_download(
            repo_id=args.router_weight_path,
            filename=f"step{i}_fe_weight.pth",
            repo_type="model",
        )

        router_path = hf_hub_download(
            repo_id=args.router_weight_path,
            filename=f"step{i}_router_weight.pth",
            repo_type="model",
        )

        fe_weight = torch.load(fe_path, map_location="cpu").to(model_dtype)
        classifier_weight = torch.load(router_path, map_location="cpu").transpose(0, 1).to(model_dtype)
        
        lora_id = 0
        for lora_path in inference_model_path[:(i+1)]:
            model.load_adapter(
                peft_model_id=args.base_path,
                adapter_name=f"{lora_id}",
                adapter_kwargs={
                    "subfolder": lora_path,
                },
            )
            lora_id += 1

        model.model.moe_classifier.weight = torch.nn.Parameter(classifier_weight)
        model.model.fe.weight = torch.nn.Parameter(fe_weight)        
        model.to(device)

        cur_inference_tasks = inference_tasks[0:i+1]
        for inference_task_id in range(len(cur_inference_tasks)):
            inference_task = inference_tasks[inference_task_id]
            # Prepare the data
            if args.benchmark == "non-executable":
                train, test, infer_dataset = create_codetask_dataset(
                    inference_task,
                    args.seed,
                    -1,
                    -1,
                    -1
                )
            else:
                train, test, infer_dataset = create_executable_dataset(
                    inference_task,
                    args.seed,
                    -1,
                    -1,
                    -1
                )

            infer_dataset = infer_dataset



            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=int(args.max_prompt_len[i]),
                max_ans_len=int(args.max_ans_len[i]),
                pad_to_multiple_of=8,
                inference=True
            )
            infer_sampler = SequentialSampler(infer_dataset)
            infer_dataloader = DataLoader(infer_dataset,
                                            collate_fn=inf_data_collator,
                                            # sampler=infer_sampler,
                                            shuffle=True,
                                            batch_size=args.inference_batch,)
            
            # default the LLM is decoder only model, so padding side is left
            assert tokenizer.padding_side == 'left'
            assert tokenizer.truncation_side == "left"

            # inference_model_path = args.inference_model_path
            
            # Inference !
            print(f"***** Start inference of step {i}: task {inference_task}*****")
            sources_sequences, predicted_sequences, ground_truths = prediction(model, tokenizer, inference_task, infer_dataloader, device, generation_config, max_ans_len=int(args.max_ans_len[i]))
            # save_inference_results({}, sources_sequences, predicted_sequences, ground_truths, i, inference_task)
            
            # Get BLEU/SmoothBLEU
            # - For hf:* tasks: compute pure-text metrics.
            #   * hf:CodeSearchNet, hf:TheVault_Csharp => smooth_bleu
            #   * other hf:* => bleu
            # - For legacy TRACE tasks: no longer supported here (removed).
            if args.benchmark == "non-executable":
                evaluation_result = _task_eval_from_predictions(inference_task, sources_sequences, predicted_sequences, ground_truths)
            else:
                evaluation_result = {}

            # if args.global_rank <= 0:  # only one process is running
            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, i, inference_task)
    
if __name__ == "__main__":
    main()
