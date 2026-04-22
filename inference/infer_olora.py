#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from transformers import AutoModelForCausalLM

from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import load_hf_tokenizer, set_random_seed, to_device
from training.params import AllDatasetName
from evaluations import (
    eval_20Minuten,
    eval_CStance,
    eval_FOMC,
    eval_MeetingBank,
    eval_NumGLUE_cm,
    eval_NumGLUE_ds,
    eval_PapyrusF,
    eval_Py150,
    eval_ScienceQA,
)


TASK_EVAL_FN = {
    "ScienceQA": eval_ScienceQA.eval,
    "MeetingBank": eval_MeetingBank.eval,
    "C-STANCE": eval_CStance.eval,
    "Papyrus-f": eval_PapyrusF.eval,
    "Py150": eval_Py150.eval,
    "FOMC": eval_FOMC.eval,
    "NumGLUE-cm": eval_NumGLUE_cm.eval,
    "NumGLUE-ds": eval_NumGLUE_ds.eval,
}


def parse_args():
    def list_of_strings(arg: str):
        return arg.split(',')

    parser = argparse.ArgumentParser(
        description="Inference script for O-LoRA (without Any-SSR router/MoE dependencies)."
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path that contains per-task datasets (train/eval/test json).')
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/',
                        help='Path for cached dataset tensors.')

    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Base model path or HF model id.')

    parser.add_argument(
        '--adapter_path',
        type=str,
        default=None,
        help='Single O-LoRA adapter path used for all tasks.'
    )
    parser.add_argument(
        '--adapter_paths',
        type=str,
        default=None,
        help='Comma-separated adapter paths aligned with inference_tasks order. '
             'If a single path is provided, it is reused for all tasks.'
    )

    parser.add_argument('--inference_tasks', type=list_of_strings, default='all',
                        help='Comma-separated task names, or "all".')
    parser.add_argument('--inference_batch', type=int, default=1,
                        help='Inference batch size.')

    parser.add_argument('--max_prompt_len', type=int, default=512,
                        help='Maximum prompt length.')
    parser.add_argument('--max_ans_len', type=int, default=256,
                        help='Maximum generated answer length.')

    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference_output_path', type=str, required=True,
                        help='Directory to save json results.')
    return parser.parse_args()


def parse_adapter_list(args, tasks: List[str]) -> Dict[str, str]:
    if args.adapter_path is None and args.adapter_paths is None:
        raise ValueError('Please provide either --adapter_path or --adapter_paths.')

    if args.adapter_path is not None and args.adapter_paths is not None:
        raise ValueError('Use only one of --adapter_path or --adapter_paths.')

    if args.adapter_path is not None:
        return {task: args.adapter_path for task in tasks}

    adapter_paths = [p.strip() for p in args.adapter_paths.split(',') if p.strip()]
    if len(adapter_paths) == 1:
        return {task: adapter_paths[0] for task in tasks}

    if len(adapter_paths) != len(tasks):
        raise ValueError(
            f'Number of adapter paths ({len(adapter_paths)}) must match number of tasks ({len(tasks)}), '
            'or pass a single adapter path.'
        )

    return {task: adapter_paths[i] for i, task in enumerate(tasks)}


def load_base_and_adapter(base_model_path: str, adapter_path: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        trust_remote_code=True,
    )

    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    except Exception:
        # Fallback for environments where model has native adapter loading hooks.
        if hasattr(model, 'load_adapter'):
            model.load_adapter(adapter_path)
        else:
            raise

    model.to(device)
    model.eval()
    return model


def run_prediction(model, tokenizer, infer_dataloader, device, args):
    predicted_sequences = []
    sources_sequences = []
    ground_truths = []

    progress_bar = tqdm(total=len(infer_dataloader), leave=True)

    for step, batch in enumerate(infer_dataloader):
        sources_sequences += batch['sources']
        ground_truths += batch['gts']

        del batch['sources']
        del batch['gts']

        batch = to_device(batch, device)
        prompt_len = batch['input_ids'].shape[1]

        progress_bar.update(1)
        progress_bar.set_description(f"Step {step}", refresh=False)

        with torch.no_grad():
            generate_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=args.max_ans_len,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_return_sequences=1,
                use_cache=True,
            )

        sequences = tokenizer.batch_decode(
            generate_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        predicted_sequences += sequences

    return sources_sequences, predicted_sequences, ground_truths


def evaluate_task(task: str, sources_sequences: List[str], predicted_sequences: List[str], ground_truths: List[str]):
    if task == '20Minuten':
        return eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)

    eval_fn = TASK_EVAL_FN.get(task)
    if eval_fn is None:
        return {}
    return eval_fn(predicted_sequences, ground_truths)


def save_inference_results(output_path: str,
                           task: str,
                           adapter_path: str,
                           evaluation_result: dict,
                           sources_sequences: list,
                           predicted_sequences: list,
                           ground_truths: list):
    os.makedirs(output_path, exist_ok=True)
    payload = {
        'task': task,
        'adapter_path': adapter_path,
        'eval': evaluation_result,
        'prompts': sources_sequences,
        'results': predicted_sequences,
        'labels': ground_truths,
    }
    with open(os.path.join(output_path, f'results-{task}.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.inference_tasks[0] == 'all':
        tasks = AllDatasetName
    else:
        tasks = args.inference_tasks

    adapter_map = parse_adapter_list(args, tasks)

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == 'left'

    for task in tasks:
        adapter_path = adapter_map[task]
        print(f'***** Start inference: task={task}, adapter={adapter_path} *****')

        dataset_path = os.path.join(args.data_path, task)
        _, _, test_dataset = create_prompt_dataset(
            -1,
            dataset_path,
            args.data_output_path,
            args.seed,
            distributed=False,
        )

        inf_data_collator = DataCollator(
            tokenizer,
            padding='longest',
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True,
        )

        infer_sampler = SequentialSampler(test_dataset)
        infer_dataloader = DataLoader(
            test_dataset,
            collate_fn=inf_data_collator,
            sampler=infer_sampler,
            batch_size=args.inference_batch,
        )

        model = load_base_and_adapter(args.model_name_or_path, adapter_path, device)

        sources_sequences, predicted_sequences, ground_truths = run_prediction(
            model, tokenizer, infer_dataloader, device, args
        )

        evaluation_result = evaluate_task(task, sources_sequences, predicted_sequences, ground_truths)
        print(f'[{task}] Eval: {evaluation_result}')

        save_inference_results(
            args.inference_output_path,
            task,
            adapter_path,
            evaluation_result,
            sources_sequences,
            predicted_sequences,
            ground_truths,
        )


if __name__ == '__main__':
    main()
