"""
Load baseline prediction JSON files from HuggingFace and compute evaluation metrics.

Repo  : dongg18/LoRa_PerTask_CodeT5-770m  (model repo, branch: main)
Folder: output/

Task policy:
  - CodeSearchNet, TheVault_Csharp  -> smooth BLEU  (calc_codebleu=False)
  - All other tasks                 -> CodeBLEU     (calc_codebleu=True)

Run:
  python eval_baselines_from_hf.py [--output_dir ./hf_eval_results]
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import hf_hub_download
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG

# --------------------------------------------------------------------------- #
# Repo configuration
# --------------------------------------------------------------------------- #
REPO_ID   = "dongg18/LoRa_PerTask_CodeT5-700m_without_pool"
REPO_TYPE = "model"

HF_FILES = [
    "output/BFP_test_preds.json",
    "output/CONCODE_test_preds.json",
    "output/CoST_test_preds.json",
    "output/CodeSearchNet_test_preds.json",
    "output/CodeTrans_test_preds.json",
    "output/RunBugRun_test_preds.json",
    "output/TheVault_Csharp_test_preds.json",
]

# Tasks that use smooth BLEU; all others use CodeBLEU
SMOOTH_BLEU_TASKS = {"CodeSearchNet", "TheVault_Csharp"}
# --------------------------------------------------------------------------- #


def load_json_from_hf(filename: str) -> dict:
    """Download a single JSON file from HF Hub and parse it."""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
    )
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_file(data: dict) -> dict:
    """Compute metrics for one baseline result file."""
    summary = data.get("summary", {})
    task = summary.get("task", "unknown")
    predictions_list = data.get("predictions", [])

    if not predictions_list:
        raise ValueError(f"No predictions found for task '{task}'.")

    preds = [item["prediction"] for item in predictions_list]
    refs  = [item["ground-truth"] for item in predictions_list]

    if task in SMOOTH_BLEU_TASKS:
        calc_codebleu = False
        language = None
    else:
        calc_codebleu = True
        language = DATASET_TO_OUTPUT_LANG.get(task)
        if language is None:
            print(
                f"[WARNING] Task '{task}' not found in DATASET_TO_OUTPUT_LANG. "
                f"Falling back to smooth BLEU."
            )
            calc_codebleu = False

    metrics = compute_metrics(
        predictions=preds,
        references=refs,
        calc_codebleu=calc_codebleu,
        language=language,
    )

    return {
        "task":        task,
        "model":       summary.get("model", "unknown"),
        "lora_path":   summary.get("lora_path", ""),
        "split":       summary.get("split", "test"),
        "n_examples":  len(preds),
        "metric_mode": "codebleu" if calc_codebleu else "smooth_bleu",
        "metrics":     metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRa_PerTask_CodeT5-770m baselines from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        default="./hf_eval_results",
        help="Directory to write per-file metric JSONs (default: ./hf_eval_results).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Repo : {REPO_ID}")
    print(f"Files: {len(HF_FILES)}\n")

    all_results = []

    for filename in HF_FILES:
        print(f"Processing: {filename}")
        try:
            data   = load_json_from_hf(filename)
            result = evaluate_file(data)

            print(
                f"  Task: {result['task']} | Model: {result['model']} | "
                f"Mode: {result['metric_mode']} | N: {result['n_examples']}"
            )
            print(f"  Metrics: {result['metrics']}")

            out_name = os.path.splitext(os.path.basename(filename))[0] + "_metrics.json"
            out_path = os.path.join(args.output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved -> {out_path}\n")

            all_results.append(result)

        except Exception as e:
            print(f"  [ERROR] Failed to process '{filename}': {e}\n")

    agg_path = os.path.join(args.output_dir, "all_results.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {agg_path}")


if __name__ == "__main__":
    main()
