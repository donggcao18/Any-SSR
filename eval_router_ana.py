import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from tqdm import tqdm

from torch.utils.data import DataLoader, SequentialSampler

from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.data.raw_datasets import CODETASK_TASKS

import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate continual router for Any-SSR")
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
        help="Number of decoder blocks used as feature extractor (must match training)",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=10000,
        help="Feature projection dimension (must match training)",
    )
    parser.add_argument(
        "--router_weights_path",
        type=str,
        default=os.environ.get(
            "ANYSSR_ROUTER_WEIGHTS_PATH", os.path.join("output_models", "router_weights")
        ),
        help="Directory containing saved router weight checkpoints",
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
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_ans_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Ordered task list. Use 'hf:<name>' for HuggingFace tasks or a plain "
            "name for local datasets. Defaults to all 8 CODETASK tasks."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help=(
            "Path to a log file. Results are always printed to stdout; "
            "this additionally writes them to the specified file."
        ),
    )
    parser.add_argument(
        "--log_predictions_jsonl",
        type=str,
        default=None,
        help=(
            "If set, writes per-sample router predictions to this JSONL file (one record per sample)."
        ),
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
        help=(
            "When using --log_predictions_jsonl, also log weight/feature summary stats for pred/gt classes."
        ),
    )
    return parser.parse_args()


def _get_backbone(model: AutoModelForCausalLM):
    """Return the decoder stack module (varies across architectures)."""
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
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
    if feature_layers is None or feature_layers <= 0:
        return

    if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
        backbone.layers = nn.ModuleList(list(backbone.layers)[:feature_layers])
        return

    if hasattr(backbone, "h") and isinstance(backbone.h, nn.ModuleList):
        backbone.h = nn.ModuleList(list(backbone.h)[:feature_layers])
        return

    raise ValueError(
        f"Backbone {backbone.__class__.__name__} doesn't expose .layers or .h for truncation."
    )


class RouterFeatureExtractorWithHead(nn.Module):
    """Eval-time router: x = fe(mean_hidden); logits = cls_head(x)."""

    def __init__(self, base_lm: AutoModelForCausalLM, gamma: int, feature_layers: int, n_tasks: int):
        super().__init__()
        self.base_lm = base_lm

        backbone = _get_backbone(self.base_lm)
        _truncate_decoder_layers(backbone, feature_layers)

        hidden_size = _get_hidden_size(self.base_lm)
        self.fe = nn.Linear(in_features=hidden_size, out_features=gamma, bias=True)
        self.cls_head = nn.Linear(in_features=gamma, out_features=n_tasks, bias=False)

    def forward(self, input_ids: torch.LongTensor):
        outputs = self.base_lm(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]

        hidden_mean = hidden_states.mean(dim=1).to(
            device=self.fe.weight.device, dtype=self.fe.weight.dtype
        )
        x = self.fe(hidden_mean)
        return self.cls_head(x)

    def extract_features(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return x = fe(mean_hidden) as float32 (for logging)."""
        outputs = self.base_lm(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]
        hidden_mean = hidden_states.mean(dim=1).to(
            device=self.fe.weight.device, dtype=self.fe.weight.dtype
        )
        x = self.fe(hidden_mean)
        return x.detach().to(torch.float32)


def load_model_and_tokenizer(step: int, args):
    base_lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda:0",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = RouterFeatureExtractorWithHead(
        base_lm=base_lm,
        gamma=args.gamma,
        feature_layers=args.feature_layers,
        n_tasks=step + 1,
    )

    return model, tokenizer


def train(args):
    inference_tasks = args.tasks
    router_weights_path = args.router_weights_path
    dataset_path = args.dataset_path
    dataset_cache_path = args.dataset_cache_path

    logger = logging.getLogger("eval_router")
    step_results = []  # list of (step, tasks_seen, correct, total, acc)

    pred_f = None
    if args.log_predictions_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_predictions_jsonl)), exist_ok=True)
        pred_f = open(args.log_predictions_jsonl, "a", encoding="utf-8")

    def _write_pred(rec: dict):
        if pred_f is None:
            return
        pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pred_f.flush()

    def eval_router(model: RouterFeatureExtractorWithHead, infer_dataloader, step):
        model_dtype = next(model.parameters()).dtype

        fe_weight = torch.load(
            f"{router_weights_path}/step{step}_fe_weight.pth",
            map_location=model.fe.weight.device,
        ).to(model_dtype)
        classifier_weight = torch.load(
            f"{router_weights_path}/step{step}_router_weight.pth",
            map_location=model.cls_head.weight.device,
        ).transpose(0, 1).to(model_dtype)

        model.cls_head.weight = torch.nn.Parameter(classifier_weight)
        model.fe.weight = torch.nn.Parameter(fe_weight)

        W = model.cls_head.weight.detach().to(torch.float32)  # [num_classes, gamma]

        with torch.no_grad():
            count = 0
            correct = 0
            tasks_seen = inference_tasks[: step + 1]
            logger.info(f"Step {step} | Tasks: {tasks_seen}")
            logger.info(f"-" * 60)

            for _, batch in enumerate(infer_dataloader):
                labels = batch["gts"]
                input_ids = batch["input_ids"].to("cuda")

                logits = model(input_ids).to(torch.float32)  # [bs, n_tasks]

                pred_id = logits.argmax(dim=-1).item()
                gt_id = int(labels[0])
                is_correct = gt_id == pred_id

                if is_correct:
                    correct += 1
                else:
                    logger.info(
                        f"  [WRONG] sample={count} "
                        f"pred={pred_id} ({inference_tasks[pred_id]}) "
                        f"gt={gt_id} ({inference_tasks[gt_id]})"
                    )

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
                            x = model.extract_features(input_ids)[0]  # [gamma]

                            def _vec_stats(v: torch.Tensor):
                                return {
                                    "l2": float(v.norm().item()),
                                    "mean": float(v.mean().item()),
                                    "std": float(v.std(unbiased=False).item()),
                                    "abs_mean": float(v.abs().mean().item()),
                                }

                            w_pred = W[pred_id]
                            w_gt = W[gt_id]

                            logit_pred = float(logits[0, pred_id].item())
                            logit_gt = float(logits[0, gt_id].item())

                            rec["features"] = {
                                "x": _vec_stats(x),
                                "x_norm": float(x.norm().item()),
                            }
                            rec["weights"] = {
                                "pred": {
                                    "id": int(pred_id),
                                    "task": inference_tasks[pred_id],
                                    **_vec_stats(w_pred),
                                    "cos_wx": float(
                                        torch.dot(w_pred, x).item()
                                        / (
                                            w_pred.norm().item()
                                            * (x.norm().item() + 1e-12)
                                            + 1e-12
                                        )
                                    ),
                                    "dot_wx": float(torch.dot(w_pred, x).item()),
                                    "logit": logit_pred,
                                },
                                "gt": {
                                    "id": int(gt_id),
                                    "task": inference_tasks[gt_id],
                                    **_vec_stats(w_gt),
                                    "cos_wx": float(
                                        torch.dot(w_gt, x).item()
                                        / (
                                            w_gt.norm().item()
                                            * (x.norm().item() + 1e-12)
                                            + 1e-12
                                        )
                                    ),
                                    "dot_wx": float(torch.dot(w_gt, x).item()),
                                    "logit": logit_gt,
                                },
                            }

                        _write_pred(rec)

                count += 1

            acc = correct / max(count, 1)
            logger.info(f"Step {step} | correct={correct}/{count} | acc={acc:.4f}")
            step_results.append((step, tasks_seen[:], correct, count, acc))

    for i in range(0, len(inference_tasks)):
        model, tokenizer = load_model_and_tokenizer(i, args)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        cur_inference_tasks = inference_tasks[0 : i + 1]
        all_datasets = []
        for inference_task_id in range(len(cur_inference_tasks)):
            inference_task = inference_tasks[inference_task_id]
            if isinstance(inference_task, str) and inference_task.startswith("hf:"):
                cur_dataset_path = inference_task
            else:
                cur_dataset_path = os.path.join(dataset_path, inference_task)

            train_ds, eval_ds, infer_dataset = create_prompt_dataset(
                -1,
                cur_dataset_path,
                dataset_cache_path,
                42,
                distributed=False,
            )

            infer_dataset.answer_dataset = [
                inference_task_id for _ in infer_dataset.answer_dataset
            ]
            all_datasets.append(infer_dataset)

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

        eval_router(model, infer_dataloader, i)

    logger.info("=" * 60)
    logger.info("ROUTER EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Step':<6} {'#Tasks':<8} {'Correct':<10} {'Total':<10} {'Acc':<10}")
    logger.info("-" * 60)
    for step, tasks_seen, correct, total, acc in step_results:
        logger.info(
            f"{step:<6} {len(tasks_seen):<8} {correct:<10} {total:<10} {acc:<10.4f}"
        )
    logger.info("=" * 60)
    if step_results:
        avg_acc = sum(r[4] for r in step_results) / len(step_results)
        logger.info(f"Average accuracy across all steps: {avg_acc:.4f}")

    if pred_f is not None:
        pred_f.close()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

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