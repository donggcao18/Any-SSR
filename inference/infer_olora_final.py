#!/usr/bin/env python
"""
Standalone seen-task inference for saved O-LoRA checkpoints.

When --checkpoint_path is provided, this evaluates every checkpoint folder from
--start_task_id to --end_task_id. Checkpoint i is evaluated on all seen tasks
0..i, and the normal metrics/predictions are saved for reporting.
"""
import argparse
import copy
import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM

from model.base_model import CL_Base_Model
from training.params import AllDatasetName
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_codetask_dataset
from utils.utils import load_hf_tokenizer


# Same order as AllDatasetName:
# [CONCODE, CodeTrans, CodeSearchNet, BFP, KodCode, RunBugRun, TheVault_Csharp, CoST]
MAX_PROMPT_LENS = [320, 320, 256, 130, 512, 256, 256, 256]
MAX_ANS_LENS = [150, 256, 128, 120, 300, 128, 128, 128]


def merge_lora_checkpoint(state_dict: dict, lora_alpha: int, lora_r: int) -> dict:
    """
    Convert a PEFT/O-LoRA state dict into a plain HuggingFace state dict by
    merging LoRA deltas into the base weights.
    """
    scaling = lora_alpha / lora_r
    hf_state = {}

    for key, value in state_dict.items():
        hf_key = key.replace("base_model.model.", "", 1)

        if ".base_layer.weight" in hf_key:
            hf_state[hf_key.replace(".base_layer.weight", ".weight")] = value.clone()
        elif ".base_layer.bias" in hf_key:
            hf_state[hf_key.replace(".base_layer.bias", ".bias")] = value.clone()
        elif any(tag in hf_key for tag in (
            ".lora_A.",
            ".lora_B.",
            ".loranew_A.",
            ".loranew_B.",
            "lora_dropout.",
            "lora_embedding_",
        )):
            continue
        else:
            hf_state[hf_key] = value.clone()

    for key, value in state_dict.items():
        for a_tag in (".lora_A.", ".loranew_A."):
            if a_tag not in key:
                continue

            b_tag = a_tag.replace("_A.", "_B.")
            b_key = key.replace(a_tag, b_tag)
            if b_key not in state_dict:
                continue

            lora_a = value.float()
            lora_b = state_dict[b_key].float()
            delta = (lora_b @ lora_a) * scaling

            layer_prefix = key.split(a_tag)[0]
            hf_weight_key = layer_prefix.replace("base_model.model.", "", 1) + ".weight"
            if hf_weight_key in hf_state:
                hf_state[hf_weight_key] = hf_state[hf_weight_key].float() + delta
            else:
                print(
                    f"[WARN] no base weight for adapter key '{key}' "
                    f"(tried '{hf_weight_key}')"
                )

    return hf_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Single saved round dir, for example .../7.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Root directory containing checkpoint folders 0, 1, ..., 7.",
    )
    parser.add_argument(
        "--start_task_id",
        type=int,
        default=0,
        help="First checkpoint id to evaluate when --checkpoint_path is used.",
    )
    parser.add_argument(
        "--end_task_id",
        type=int,
        default=7,
        help="Last checkpoint id to evaluate when --checkpoint_path is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-Coder-1.5B",
        help="Base model identifier used for architecture/tokenizer.",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument(
        "--num_test",
        type=int,
        default=-1,
        help="Number of test examples per task (-1 = all).",
    )
    parser.add_argument(
        "--seen_task_count",
        type=int,
        default=None,
        help="For single-checkpoint mode, evaluate only the first N tasks.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write predictions and metrics.",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_path is None:
        parser.error("one of --checkpoint or --checkpoint_path is required")
    if args.checkpoint is not None and args.checkpoint_path is not None:
        parser.error("use either --checkpoint or --checkpoint_path, not both")
    return args


def set_inference_defaults(args):
    args.local_rank = -1
    args.global_rank = 0
    args.do_sample = False
    args.temperature = None
    args.top_p = None
    args.repetition_penalty = 1.0
    args.zero_stage = 0
    args.max_ans_len = MAX_ANS_LENS


def load_merged_model(checkpoint, args, dtype, device):
    adapter_file = os.path.join(checkpoint, "adapter_model.bin")
    full_ckpt_file = os.path.join(checkpoint, "pytorch_model.bin")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
    )
    model.config.use_cache = True

    if os.path.isfile(adapter_file):
        print(f"Loading PEFT adapter from: {adapter_file}")
        adapter_state_dict = torch.load(adapter_file, map_location="cpu")
        base_state_dict = {key: value for key, value in model.state_dict().items()}
        scaling = args.lora_alpha / args.lora_r

        for key, value in adapter_state_dict.items():
            for a_tag in (".lora_A.", ".loranew_A."):
                if a_tag not in key:
                    continue

                b_tag = a_tag.replace("_A.", "_B.")
                b_key = key.replace(a_tag, b_tag)
                if b_key not in adapter_state_dict:
                    continue

                lora_a = value.float()
                lora_b = adapter_state_dict[b_key].float()
                delta = (lora_b @ lora_a) * scaling

                layer_prefix = key.split(a_tag)[0]
                hf_weight_key = layer_prefix.replace("base_model.model.", "", 1) + ".weight"
                if hf_weight_key in base_state_dict:
                    base_state_dict[hf_weight_key] = (
                        base_state_dict[hf_weight_key].float() + delta
                    )
                else:
                    print(
                        f"[WARN] no base weight for adapter key '{key}' "
                        f"(tried '{hf_weight_key}')"
                    )

        hf_state_dict = {key: value.to(dtype) for key, value in base_state_dict.items()}
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    elif os.path.isfile(full_ckpt_file):
        print(f"Loading full checkpoint from: {full_ckpt_file}")
        raw_state_dict = torch.load(full_ckpt_file, map_location="cpu")
        if any(key.startswith("base_model.model.") for key in raw_state_dict):
            print("Detected PEFT/O-LoRA state dict; merging LoRA weights into base.")
            hf_state_dict = merge_lora_checkpoint(raw_state_dict, args.lora_alpha, args.lora_r)
        else:
            print("Detected plain HF state dict; loading directly.")
            hf_state_dict = raw_state_dict

        hf_state_dict = {key: value.to(dtype) for key, value in hf_state_dict.items()}
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    else:
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint}'. "
            "Expected 'adapter_model.bin' or 'pytorch_model.bin'."
        )

    if missing:
        print(f"[WARN] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model.to(device)
    model.eval()
    print("Model ready on", device)
    return model


def build_seen_task_loaders(tokenizer, args, seen_task_count):
    if seen_task_count < 1 or seen_task_count > len(AllDatasetName):
        raise ValueError(
            f"seen_task_count must be between 1 and {len(AllDatasetName)}, "
            f"got {seen_task_count}"
        )

    test_task_list = {}
    for task_idx, dataset in enumerate(AllDatasetName[:seen_task_count]):
        _, _, test_dataset = create_codetask_dataset(
            dataset,
            args.seed,
            num_train=-1,
            num_eval=-1,
            num_test=args.num_test,
        )
        collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=MAX_PROMPT_LENS[task_idx],
            max_ans_len=MAX_ANS_LENS[task_idx],
            pad_to_multiple_of=8,
            inference=True,
        )
        test_task_list[dataset] = DataLoader(
            test_dataset,
            collate_fn=collator,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.per_device_eval_batch_size,
        )
        print(f"  [{task_idx}] {dataset}: {len(test_dataset)} test examples")
    return test_task_list


def evaluate_checkpoint(checkpoint, output_dir, seen_task_count, base_args, dtype, device):
    args = copy.copy(base_args)
    args.checkpoint = checkpoint
    args.output_dir = output_dir
    args.seen_task_count = seen_task_count
    set_inference_defaults(args)

    print(f"***** Evaluating checkpoint {checkpoint} on seen tasks 0..{seen_task_count - 1} *****")
    tokenizer = load_hf_tokenizer(checkpoint, fast_tokenizer=True)
    model = load_merged_model(checkpoint, args, dtype, device)
    test_task_list = build_seen_task_loaders(tokenizer, args, seen_task_count)

    trainer = CL_Base_Model(
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        train_task_list={},
        eval_task_list={},
        test_task_list=test_task_list,
        args=args,
    )
    trainer.test_all_tasks_and_save_predictions()

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_path is not None:
        if args.output_dir is None:
            args.output_dir = os.path.join(os.path.abspath(args.checkpoint_path), "seen_task_eval")

        for task_id in range(args.start_task_id, args.end_task_id + 1):
            checkpoint = os.path.join(args.checkpoint_path, str(task_id))
            if not os.path.isdir(checkpoint):
                print(f"[WARN] skipping missing checkpoint: {checkpoint}")
                continue

            evaluate_checkpoint(
                checkpoint=checkpoint,
                output_dir=os.path.join(args.output_dir, f"after_task_{task_id}"),
                seen_task_count=task_id + 1,
                base_args=args,
                dtype=dtype,
                device=device,
            )
        return

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        seen_task_count=args.seen_task_count or len(AllDatasetName),
        base_args=args,
        dtype=dtype,
        device=device,
    )


if __name__ == "__main__":
    main()
