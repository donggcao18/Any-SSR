import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

from torch.utils.data import DataLoader, SequentialSampler

from utils.data.data_utils import create_prompt_dataset
from utils.data.raw_datasets import CODETASK_TASKS
from utils.data.data_collator import DataCollator

import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train continual router for Any-SSR")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local path (Qwen/LLaMA/Mistral/etc.)",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value (e.g. '0', '0,1')",
    )
    parser.add_argument(
        "--feature_layers",
        type=int,
        default=4,
        help="Number of backbone decoder blocks to use as feature extractor",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=10000,
        help="Output dimension of the feature projection layer",
    )
    parser.add_argument(
        "--router_weights_path",
        type=str,
        default=os.environ.get(
            "ANYSSR_ROUTER_WEIGHTS_PATH", os.path.join("output_models", "router_weights")
        ),
        help="Directory to save router weight checkpoints",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get(
            "ANYSSR_DATASET_PATH",
            os.path.join("dataset", "TRACE-Benchmark", "LLM-CL-Benchmark_5000"),
        ),
        help="Root directory for local datasets",
    )
    parser.add_argument(
        "--dataset_cache_path",
        type=str,
        default=os.environ.get(
            "ANYSSR_DATASET_CACHE_PATH",
            os.path.join("output_models", "outputs_router_dataset_cache"),
        ),
        help="Directory for dataset cache",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="Maximum prompt token length for the data collator",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="Maximum answer token length for the data collator",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="DataLoader batch size")
    parser.add_argument(
        "--rls_lambda",
        type=float,
        default=100.0,
        help="Regularisation coefficient for the initial RLS matrix inversion",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs= "+",
        default=None,
        help=(
            "Ordered list of task identifiers. Use 'hf:<name>' for HuggingFace datasets "
            "or a plain name for local datasets under --dataset_path. "
            "Defaults to the 8 coding tasks if omitted."
        ),
    )
    return parser.parse_args()


def _get_backbone(model: AutoModelForCausalLM):
    """Return the decoder stack module (varies across architectures)."""
    # LLaMA/Mistral/DeepSeek/Qwen2 style
    if hasattr(model, "model"):
        return model.model
    # GPTNeoX style
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    # GPT2 style
    if hasattr(model, "transformer"):
        return model.transformer
    raise ValueError(
        f"Unsupported model backbone for {model.__class__.__name__}; can't locate decoder stack."
    )


def _get_hidden_size(model: AutoModelForCausalLM) -> int:
    if hasattr(model.config, "hidden_size") and model.config.hidden_size is not None:
        return int(model.config.hidden_size)
    if hasattr(model.config, "n_embd") and model.config.n_embd is not None:
        return int(model.config.n_embd)
    raise ValueError(
        f"Can't infer hidden size from config class {model.config.__class__.__name__}."
    )


def _truncate_decoder_layers(backbone: nn.Module, feature_layers: int):
    """Keep only the first N layers of the decoder stack."""
    if feature_layers is None or feature_layers <= 0:
        return

    if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
        backbone.layers = nn.ModuleList(list(backbone.layers)[:feature_layers])
        return

    # GPT2-style
    if hasattr(backbone, "h") and isinstance(backbone.h, nn.ModuleList):
        backbone.h = nn.ModuleList(list(backbone.h)[:feature_layers])
        return

    raise ValueError(
        f"Backbone {backbone.__class__.__name__} doesn't expose .layers or .h for truncation."
    )


class RouterFeatureExtractor(nn.Module):
    """Architecture-agnostic feature extractor used by analytic router.

    It runs the base causal LM forward pass and returns x = fe(mean_hidden).
    """

    def __init__(self, base_lm: AutoModelForCausalLM, gamma: int, feature_layers: int = 4):
        super().__init__()
        self.base_lm = base_lm

        backbone = _get_backbone(self.base_lm)
        _truncate_decoder_layers(backbone, feature_layers)

        hidden_size = _get_hidden_size(self.base_lm)
        self.fe = nn.Linear(in_features=hidden_size, out_features=gamma)

    def forward(self, input_ids: torch.LongTensor):
        outputs = self.base_lm(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # Prefer last_hidden_state if present; else fall back to tuple index 0.
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]

        hidden_mean = hidden_states.mean(dim=1)
        return self.fe(hidden_mean)


def load_model_and_tokenizer(args):
    base_lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = RouterFeatureExtractor(
        base_lm=base_lm,
        gamma=args.gamma,
        feature_layers=args.feature_layers,
    )

    return model, tokenizer


def train(args):
    model, tokenizer = load_model_and_tokenizer(args)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    router_weights_path = args.router_weights_path
    dataset_path = args.dataset_path
    dataset_cache_path = args.dataset_cache_path

    # Task order (must match your downstream inference order)
    inference_tasks = args.tasks

    def train_initial_router(model, infer_dataloader, step):
        """Train first tasks"""
        with torch.no_grad():
            print(
                f"---start training from {inference_tasks[0]} to {inference_tasks[step]}"
            )
            count = 0
            print("-----------------------start training-------------------")

            pbar = tqdm(infer_dataloader, desc=f"[Step {step}] Initial training", unit="batch")
            for steps, batch in enumerate(pbar):
                labels = batch["gts"]
                input_ids = batch["input_ids"].to("cuda")

                new_activation = model(input_ids).to(torch.float32)
                labels = torch.tensor(labels)
                label_onehot = F.one_hot(labels, step + 1).float().to("cuda")

                if count == 0:
                    auto_cor = torch.t(new_activation) @ new_activation
                    crs_cor = torch.t(new_activation) @ (label_onehot)
                else:
                    auto_cor += torch.t(new_activation) @ new_activation
                    crs_cor += torch.t(new_activation) @ (label_onehot)

                count += 1
                pbar.set_postfix(batches=count)

            print("Calculating Reverse")

            R = np.mat(auto_cor.cpu().numpy() + args.rls_lambda * np.eye(args.gamma)).I
            R_tensor = torch.tensor(R).float().cuda(non_blocking=True).cpu()
            Delta = R_tensor @ crs_cor.cpu()

            torch.save(Delta, f"{router_weights_path}/step{step}_router_weight.pth")
            torch.save(model.fe.weight, f"{router_weights_path}/step{step}_fe_weight.pth")
            torch.save(R_tensor, f"{router_weights_path}/step{step}_R_matrix.pth")

            print(
                f"Finished initial training from {inference_tasks[0]} to {inference_tasks[step]}"
            )
            return R_tensor, Delta

    def train_subsequent_router(model, infer_dataloader, step, prev_R, prev_Delta):
        """Recursive Train"""
        prev_Delta = F.pad(prev_Delta, (0, 1), mode="constant", value=0).to("cuda")
        prev_R = prev_R.to("cuda")
        with torch.no_grad():
            print(f"Start training from {inference_tasks[0]} to {inference_tasks[step]}")
            print(f"Use step{step-1} as initial R")
            count = 0
            print("-----------------------Start Recursive Train-------------------")

            pbar = tqdm(
                infer_dataloader,
                desc=f"[Step {step}] Recursive training",
                unit="batch",
            )
            for steps, batch in enumerate(pbar):
                labels = batch["gts"]
                input_ids = batch["input_ids"].to("cuda")

                new_activation = model(input_ids).to(torch.float32)
                labels = torch.tensor(labels)
                label_onehot = F.one_hot(labels, step + 1).float().to("cuda")

                prev_R = prev_R - prev_R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.shape[0]).to("cuda")
                    + new_activation @ prev_R @ new_activation.t()
                ) @ new_activation @ prev_R
                prev_Delta = prev_Delta + prev_R @ new_activation.t() @ (
                    label_onehot - new_activation @ prev_Delta
                )
                pbar.set_postfix(batches=steps + 1)

            print("Calculate new R")
            new_R = prev_R
            new_Delta = prev_Delta

            torch.save(new_Delta, f"{router_weights_path}/step{step}_router_weight.pth")
            torch.save(model.fe.weight, f"{router_weights_path}/step{step}_fe_weight.pth")
            torch.save(new_R, f"{router_weights_path}/step{step}_R_matrix.pth")

            print(f"Finished training from {inference_tasks[0]} to {inference_tasks[step]}")
            return new_R, new_Delta

    for i in range(0, len(inference_tasks)):
        cur_inference_tasks = inference_tasks[0 : i + 1]
        all_datasets = []

        if i == 0:
            for inference_task_id in range(len(cur_inference_tasks)):
                inference_task = inference_tasks[inference_task_id]
                if isinstance(inference_task, str) and inference_task.startswith("hf:"):
                    cur_dataset_path = inference_task
                else:
                    cur_dataset_path = os.path.join(dataset_path, inference_task)

                train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
                    -1,
                    cur_dataset_path,
                    dataset_cache_path,
                    42,
                    distributed=False,
                )
                train_dataset.answer_dataset = [
                    inference_task_id for _ in train_dataset.answer_dataset
                ]
                all_datasets.append(train_dataset)
        else:
            inference_task = inference_tasks[i]
            inference_task_id = i
            if isinstance(inference_task, str) and inference_task.startswith("hf:"):
                cur_dataset_path = inference_task
            else:
                cur_dataset_path = os.path.join(dataset_path, inference_task)

            train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
                -1,
                cur_dataset_path,
                dataset_cache_path,
                42,
                distributed=False,
            )
            train_dataset.answer_dataset = [
                inference_task_id for _ in train_dataset.answer_dataset
            ]
            all_datasets.append(train_dataset)

        try:
            infer_dataset = torch.utils.data.ConcatDataset(all_datasets)
        except Exception:
            infer_dataset = all_datasets[0]

        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True,
        )
        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(
            infer_dataset,
            collate_fn=inf_data_collator,
            sampler=infer_sampler,
            batch_size=args.batch_size,
        )

        print("***** Start Training *****")
        if i == 0:
            current_R, Delta = train_initial_router(model, infer_dataloader, i)
        else:
            current_R, Delta = train_subsequent_router(
                model, infer_dataloader, i, current_R, Delta
            )


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.tasks is None:
        args.tasks = CODETASK_TASKS

    os.makedirs(args.router_weights_path, exist_ok=True)
    os.makedirs(args.dataset_cache_path, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print("-----------------------------------start training---------------------------------------")
    train(args)