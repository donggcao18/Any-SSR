"""
Load a single JSONL prediction file from HuggingFace and compute per-task
evaluation metrics using the project's existing metric code.

Repo : dongg18/PP_CodeT5-770m  (private model repo, branch: main)
File : codet5p-770m_20260421_170536/test_predictions_CoST.jsonl

JSONL schema (one JSON object per line):
{
  "task":       "CONCODE",
  "input":      "...",
  "prediction": "...",
  "label":      "..."
}

Task policy (grouped by task field):
  - CodeSearchNet, TheVault_Csharp  -> smooth BLEU  (calc_codebleu=False)
  - All other tasks                 -> CodeBLEU     (calc_codebleu=True)

Run:
  python eval_baselines_from_hf_1file.py [--output_dir ./hf_eval_results_pp]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import hf_hub_download
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG

# --------------------------------------------------------------------------- #
# Repo / file configuration
# --------------------------------------------------------------------------- #
REPO_ID   = "dongg18/PP_CodeT5-770m"
REPO_TYPE = "model"
HF_FILE   = "codet5p-770m_20260421_170536/test_predictions_CoST.jsonl"

# Datasets that use smooth BLEU; all others use CodeBLEU
SMOOTH_BLEU_TASKS = {"CodeSearchNet", "TheVault_Csharp"}
# --------------------------------------------------------------------------- #


def load_jsonl_from_hf(filename: str) -> list[dict]:
    """Download the JSONL file from HF Hub and return all parsed lines."""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
    )
    records = []
    with open(local_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARNING] Skipping malformed line {lineno}: {e}")
    return records


def evaluate_dataset_group(dataset: str, preds: list[str], refs: list[str]) -> dict:
    """Compute metrics for one dataset group."""
    if dataset in SMOOTH_BLEU_TASKS:
        calc_codebleu = False
        language = None
    else:
        calc_codebleu = True
        language = DATASET_TO_OUTPUT_LANG.get(dataset)
        if language is None:
            print(
                f"  [WARNING] Dataset '{dataset}' not in DATASET_TO_OUTPUT_LANG. "
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
        "dataset":     dataset,
        "n_examples":  len(preds),
        "metric_mode": "codebleu" if calc_codebleu else "smooth_bleu",
        "metrics":     metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PP_CodeT5-770m predictions from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        default="./hf_eval_results_pp",
        help="Directory to write metric JSONs (default: ./hf_eval_results_pp).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Repo : {REPO_ID}")
    print(f"File : {HF_FILE}\n")

    # ------------------------------------------------------------------ #
    # 1. Download and parse all lines
    # ------------------------------------------------------------------ #
    print("Downloading JSONL file...")
    records = load_jsonl_from_hf(HF_FILE)
    print(f"Loaded {len(records)} predictions.\n")

    # ------------------------------------------------------------------ #
    # 2. Log every prediction
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("ALL PREDICTIONS")
    print("=" * 70)
    for i, rec in enumerate(records):
        task         = rec.get("task", "unknown")
        ground_truth = rec.get("label", "")
        prediction   = rec.get("prediction", "")
        print(f"[{i}] Task={task}")
        print(f"     Ground-truth : {ground_truth.strip()}")
        print(f"     Prediction   : {prediction.strip()}")
        print()

    # ------------------------------------------------------------------ #
    # 3. Group by task
    # ------------------------------------------------------------------ #
    groups: dict[str, dict[str, list]] = defaultdict(lambda: {"preds": [], "refs": []})
    for rec in records:
        task         = rec.get("task", "unknown")
        ground_truth = rec.get("label", "")
        prediction   = rec.get("prediction", "")
        groups[task]["preds"].append(prediction)
        groups[task]["refs"].append(ground_truth)

    # ------------------------------------------------------------------ #
    # 4. Compute metrics per dataset
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("METRICS PER DATASET")
    print("=" * 70)
    all_results = []

    for task in sorted(groups.keys()):
        preds = groups[task]["preds"]
        refs  = groups[task]["refs"]

        result = evaluate_dataset_group(task, preds, refs)

        print(
            f"Task: {result['dataset']} | Mode: {result['metric_mode']} "
            f"| N: {result['n_examples']}"
        )
        print(f"  Metrics: {result['metrics']}\n")

        out_path = os.path.join(args.output_dir, f"{task}_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

    # ------------------------------------------------------------------ #
    # 5. Save aggregated results
    # ------------------------------------------------------------------ #
    agg_path = os.path.join(args.output_dir, "all_results.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to: {agg_path}")


if __name__ == "__main__":
    main()
