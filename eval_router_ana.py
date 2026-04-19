import os
import argparse
from transformers.models.llama import LlamaForCausalLM
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

import logging

from transformers.cache_utils import Cache, DynamicCache

from transformers.utils import logging

from transformers.models.llama import LlamaModel

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaSdpaAttention, LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaFlashAttention2, LlamaMLP, repeat_kv, apply_rotary_pos_emb

import torch.nn as nn

import torch.nn.functional as F

import logging
import transformers

from peft import LoraConfig
import torch

import json
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import get_peft_model

from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.data.raw_datasets import CODETASK_TASKS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import torch.nn.functional as F

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate continual router for Any-SSR")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="HuggingFace model name or local path")
    parser.add_argument("--cuda_devices", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0', '0,1')")
    parser.add_argument("--feature_layers", type=int, default=4,
                        help="Number of LLaMA layers used as feature extractor (must match training)")
    parser.add_argument("--gamma", type=int, default=10000,
                        help="Feature projection dimension (must match training)")
    parser.add_argument("--router_weights_path", type=str,
                        default=os.environ.get("ANYSSR_ROUTER_WEIGHTS_PATH",
                                               os.path.join("output_models", "router_weights")),
                        help="Directory containing saved router weight checkpoints")
    parser.add_argument("--dataset_path", type=str,
                        default=os.environ.get("ANYSSR_DATASET_PATH",
                                               os.path.join("dataset", "TRACE-Benchmark", "LLM-CL-Benchmark_5000")),
                        help="Root directory for local datasets")
    parser.add_argument("--dataset_cache_path", type=str,
                        default=os.environ.get("ANYSSR_DATASET_CACHE_PATH",
                                               os.path.join("output_models", "outputs_router_dataset_cache")),
                        help="Directory for dataset cache")
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_ans_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Ordered task list. Use 'hf:<name>' for HuggingFace tasks or a plain "
                             "name for local datasets. Defaults to all 8 CODETASK tasks.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to a log file. Results are always printed to stdout; "
                             "this additionally writes them to the specified file.")
    parser.add_argument(
        "--log_predictions_jsonl",
        type=str,
        default=None,
        help="If set, writes per-sample router predictions to this JSONL file (one record per sample).",
    )
    parser.add_argument(
        "--log_topk",
        type=int,
        default=5,
        help="Top-k classes to log for each sample when --log_predictions_jsonl is set.",
    )
    parser.add_argument(
        "--max_log_samples",
        type=int,
        default=0,
        help="Max number of samples to log per step (0 = log all).",
    )
    parser.add_argument(
        "--log_weight_stats",
        action="store_true",
        help="When using --log_predictions_jsonl, also log weight/feature summary stats for pred/gt classes.",
    )
    return parser.parse_args()

class NewLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, task_number, gamma, feature_layers=4):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model.layers = torch.nn.ModuleList(self.model.layers[:feature_layers])  # 仅取前4层进行分类

        # self.cls_head = torch.nn.Linear(in_features=4096, out_features=task_number)
        self.fe = torch.nn.Linear(in_features=4096, out_features=gamma)

        self.cls_head = torch.nn.Linear(in_features=gamma, out_features=task_number)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # Only takes the output of first 4 layers of the model 

        hidden_mean = outputs[0].mean(dim=1).to(device=self.fe.weight.device, dtype=self.fe.weight.dtype)

        out = self.fe(hidden_mean)

        out = self.cls_head(out)
        return out
    
    def MoeClassifier():
        pass


def load_model_and_tokenizer(step, args):
    model = NewLlamaForCausalLM.from_pretrained(
        args.model,
        device_map="cuda:0",
        torch_dtype="auto",
        task_number=step + 1,
        trust_remote_code=True,
        gamma=args.gamma,
        feature_layers=args.feature_layers,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    return model, tokenizer


def load_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    return tokenizer


def train(args):
    inference_tasks = args.tasks
    router_weights_path = args.router_weights_path
    dataset_path = args.dataset_path
    dataset_cache_path = args.dataset_cache_path
    import numpy as np

    logger = logging.getLogger("eval_router")
    step_results = []  # list of (step, tasks_seen, correct, total, acc)

    # Optional JSONL writer for per-sample prediction analysis
    pred_f = None
    if args.log_predictions_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_predictions_jsonl)), exist_ok=True)
        pred_f = open(args.log_predictions_jsonl, "a", encoding="utf-8")

    def _write_pred(rec: dict):
        if pred_f is None:
            return
        pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pred_f.flush()

    def eval_router(model, infer_dataloader, step):
        model_dtype = next(model.parameters()).dtype
        fe_weight = torch.load(f'{router_weights_path}/step{step}_fe_weight.pth', map_location=model.device).to(model_dtype)
        classifier_weight = torch.load(f'{router_weights_path}/step{step}_router_weight.pth', map_location=model.device).transpose(0, 1).to(model_dtype)
        model.cls_head.weight = torch.nn.Parameter(classifier_weight)
        model.fe.weight = torch.nn.Parameter(fe_weight)

        # cache for logging weight stats (shape: [num_classes, gamma])
        W = model.cls_head.weight.detach().to(torch.float32)
        W_n = W.norm(dim=1)

        with torch.no_grad():
            count = 0
            correct = 0
            tasks_seen = inference_tasks[:step + 1]
            logger.info(f"Step {step} | Tasks: {tasks_seen}")
            logger.info(f"-" * 60)
            for steps, batch in enumerate(infer_dataloader):
                labels = batch['gts']
                input_ids = batch['input_ids']
                input_ids = input_ids.to('cuda')

                # logits from router head (float32 for stable logging)
                logits = model(input_ids).to(torch.float32)  # [1, n_tasks]

                pred_id = logits.argmax().item()
                gt_id = int(labels[0])
                is_correct = (gt_id == pred_id)

                if is_correct:
                    correct += 1
                else:
                    logger.info(
                        f"  [WRONG] sample={count} "
                        f"pred={pred_id} ({inference_tasks[pred_id]}) "
                        f"gt={gt_id} ({inference_tasks[gt_id]})"
                    )

                # Per-sample logging (optional)
                if pred_f is not None:
                    if args.max_log_samples == 0 or count < args.max_log_samples:
                        probs = torch.softmax(logits[0], dim=-1)
                        topk = min(int(args.log_topk), probs.numel())
                        top_probs, top_ids = torch.topk(probs, k=topk)

                        rec = {
                            "step": int(step),
                            "sample": int(count),
                            "tasks_seen": tasks_seen,
                            "gt_id": gt_id,
                            "gt_task": inference_tasks[gt_id],
                            "pred_id": pred_id,
                            "pred_task": inference_tasks[pred_id],
                            "correct": bool(is_correct),
                            "prob_max": float(probs.max().item()),
                            "prob_gt": float(probs[gt_id].item()) if gt_id < probs.numel() else None,
                            "margin_pred_minus_gt": float((probs[pred_id] - probs[gt_id]).item())
                            if gt_id < probs.numel() else None,
                            "topk": [
                                {
                                    "id": int(i.item()),
                                    "task": inference_tasks[int(i.item())],
                                    "prob": float(p.item()),
                                }
                                for i, p in zip(top_ids, top_probs)
                            ],
                        }

                        if args.log_weight_stats:
                            # feature vector used for classification is x = fe(mean_hidden)
                            # we reconstruct x here to relate weights -> logits
                            # (same preprocessing as in model.forward)
                            outputs = model.model(input_ids=input_ids)
                            hidden_mean = outputs[0].mean(dim=1).to(
                                device=model.fe.weight.device, dtype=model.fe.weight.dtype
                            )
                            x = model.fe(hidden_mean).detach().to(torch.float32)[0]  # [gamma]
                            x_norm = float(x.norm().item())

                            def _vec_stats(v: torch.Tensor):
                                return {
                                    "l2": float(v.norm().item()),
                                    "mean": float(v.mean().item()),
                                    "std": float(v.std(unbiased=False).item()),
                                    "abs_mean": float(v.abs().mean().item()),
                                }

                            w_pred = W[pred_id]
                            w_gt = W[gt_id]

                            # logits are dot(W, x) (bias=False)
                            logit_pred = float(logits[0, pred_id].item())
                            logit_gt = float(logits[0, gt_id].item())

                            rec["features"] = {
                                "x": _vec_stats(x),
                                "x_norm": x_norm,
                            }
                            rec["weights"] = {
                                "pred": {
                                    "id": int(pred_id),
                                    "task": inference_tasks[pred_id],
                                    **_vec_stats(w_pred),
                                    "cos_wx": float(torch.dot(w_pred, x).item() / (w_pred.norm().item() * (x.norm().item() + 1e-12) + 1e-12)),
                                    "dot_wx": float(torch.dot(w_pred, x).item()),
                                    "logit": logit_pred,
                                },
                                "gt": {
                                    "id": int(gt_id),
                                    "task": inference_tasks[gt_id],
                                    **_vec_stats(w_gt),
                                    "cos_wx": float(torch.dot(w_gt, x).item() / (w_gt.norm().item() * (x.norm().item() + 1e-12) + 1e-12)),
                                    "dot_wx": float(torch.dot(w_gt, x).item()),
                                    "logit": logit_gt,
                                },
                            }

                        _write_pred(rec)

                count += 1

            acc = correct / count
            logger.info(
                f"Step {step} | correct={correct}/{count} | acc={acc:.4f}"
            )
            step_results.append((step, tasks_seen[:], correct, count, acc))

    # for i in range(0, len(inference_tasks) - 1):
    for i in range(0, len(inference_tasks)):
        model, tokenizer = load_model_and_tokenizer(i, args)
        tokenizer.pad_token = tokenizer.eos_token

        # cur_inference_tasks = inference_tasks[0:i+2]
        cur_inference_tasks = inference_tasks[0:i+1]
        all_datasets = []
        for inference_task_id in range(len(cur_inference_tasks)):
            inference_task = inference_tasks[inference_task_id]
            # hf:* datasets are dataset identifiers, not filesystem paths
            if isinstance(inference_task, str) and inference_task.startswith("hf:"):
                cur_dataset_path = inference_task
            else:
                cur_dataset_path = os.path.join(dataset_path, inference_task)

            # Prepare the data
            train, test, infer_dataset = create_prompt_dataset(
                -1,
                cur_dataset_path,
                dataset_cache_path,
                42,
                distributed=False
            )

            infer_dataset.answer_dataset = [inference_task_id for _ in infer_dataset.answer_dataset]
            all_datasets.append(infer_dataset)
        
        # continue
        try:
            infer_dataset = torch.utils.data.ConcatDataset(all_datasets)
        except:
            infer_dataset = all_datasets[0]

        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True,
            task=inference_task,
        )
        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(infer_dataset,
                                        collate_fn=inf_data_collator,
                                        # sampler=infer_sampler,
                                        shuffle=True,
                                        batch_size=args.batch_size)

        # Inference !
        eval_router(model, infer_dataloader, i)

    # ---- Final summary ----
    logger = logging.getLogger("eval_router")
    logger.info("=" * 60)
    logger.info("ROUTER EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Step':<6} {'#Tasks':<8} {'Correct':<10} {'Total':<10} {'Acc':<10}")
    logger.info("-" * 60)
    for step, tasks_seen, correct, total, acc in step_results:
        logger.info(f"{step:<6} {len(tasks_seen):<8} {correct:<10} {total:<10} {acc:<10.4f}")
    logger.info("=" * 60)
    if step_results:
        avg_acc = sum(r[4] for r in step_results) / len(step_results)
        logger.info(f"Average accuracy across all steps: {avg_acc:.4f}")

    if pred_f is not None:
        pred_f.close()


if __name__ == "__main__":
    args = parse_args()

    # Apply CUDA device selection before any CUDA calls
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Default task list — sourced from CODETASK_TASKS in raw_datasets.py
    if args.tasks is None:
        args.tasks = CODETASK_TASKS

    os.makedirs(args.router_weights_path, exist_ok=True)
    os.makedirs(args.dataset_cache_path, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("eval_router").info(
        "-----------------------------------start router evaluation---------------------------------------"
    )
    train(args)