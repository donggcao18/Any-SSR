import string
import json
import os
import sys
import argparse
import logging

# from rouge import rouge_scorer
# from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import compute_bleu
from collections import defaultdict

logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)
GPT2TOKENIZER = os.path.join(CURRENT_DIR, "../data/gpt2tokenizer")

DATASET_TO_OUTPUT_LANG = {'CodeTrans': 'c_sharp',
                          'CodeSearchNet': 'english',
                          'BFP': 'java',
                          'CONCODE': 'java',
                          'TheVault_Csharp': 'english',
                          'KodCode': 'python',
                          'RunBugRun': 'ruby',
                          'CoST': 'c_sharp'}

# class GPTTokenizer:
#     gpt_tokenizer = AutoTokenizer.from_pretrained(GPT2TOKENIZER, max_length=1e5)

#     def tokenize(self, s):
#         tokens = self.gpt_tokenizer.tokenize(s)
#         # GPT2 uses Byte-level BPE, which will include space as part of the word. 
#         # But for the first word of a sentence, there is no space before it. 
#         # So, we remove all the added spaces ("Ġ"). 
#         tokens = [t.lstrip("Ġ") for t in tokens]
#         return tokens


# xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


# def rouge1_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
#     else:
#         scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rouge1"].fmeasure


# def rougeL_score(prediction, ground_truth, xlingual=False):
#     if xlingual:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
#     else:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, calc_codebleu=False, language=None, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, bleu, codebleu = 0, 0, 0
    gold_map = defaultdict(list)
    pred_map = {}
    per_segment_references, translations = [], []
    pre_references, hypothesis = [], []
    for i, (pred, gold) in enumerate(zip(predictions, references)):
        gold_map[str(i)].append(smooth_bleu.splitPuncts(gold.strip().lower()))
        pred_map[str(i)] = [smooth_bleu.splitPuncts(pred.strip().lower())]
        per_segment_references.append([gold.strip().split()])
        translations.append(pred.strip().split())
        pre_references.append(gold.strip())
        hypothesis.append(pred.strip())
        gold = [gold]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    pre_references = [pre_references,]
    exact_match = 100.0 * exact_match / len(references)
    if calc_codebleu:
        bleu = compute_bleu(per_segment_references, translations, max_order=4, smooth=True)[0] * 100
        codebleu = calc_code_bleu.get_codebleu(pre_references, hypothesis, language) * 100
    else:
        bleu = smooth_bleu.bleuFromMaps(gold_map, pred_map)[0]
    metrics = {"exact_match": exact_match, "bleu": bleu, "codebleu": codebleu}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        if group in ("CodeSearchNet", "TheVault_Csharp"):
            calc_codebleu = False
            language = None
        else:    
            calc_codebleu = True
            language = DATASET_TO_OUTPUT_LANG[group]
        group_metrics = compute_metrics(task_predictions, task_references, calc_codebleu=calc_codebleu, language=language, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


if __name__ == "__main__":
    print("Running synthetic metric checks for both CodeBLEU-off and CodeBLEU-on settings...\n")

    # 1) Direct check with calc_codebleu=False (summarization-style output).
    preds_no_codebleu = [
        "returns the maximum value in the list",
        "parses a json string into an object",
    ]
    refs_no_codebleu = [
        "returns max value from the list",
        "parse a json string into object",
    ]
    metrics_no_codebleu = compute_metrics(
        preds_no_codebleu,
        refs_no_codebleu,
        calc_codebleu=False,
        language=None,
    )
    print("=== Direct compute_metrics (calc_codebleu=False) ===")
    print(metrics_no_codebleu)

    # 2) Direct check with calc_codebleu=True (code generation/translation-style output).
    preds_with_codebleu = [
        "public static int add(int a, int b) { return a + b; }",
        "public static boolean isEven(int x) { return x % 2 == 0; }",
    ]
    refs_with_codebleu = [
        "public static int add(int a, int b) { return a + b; }",
        "public static boolean isEven(int n) { return n % 2 == 0; }",
    ]
    metrics_with_codebleu = compute_metrics(
        preds_with_codebleu,
        refs_with_codebleu,
        calc_codebleu=True,
        language="java",
    )
    print("\n=== Direct compute_metrics (calc_codebleu=True, language='java') ===")
    print(metrics_with_codebleu)

    # 3) Grouped check: uses dataset policy in compute_grouped_metrics:
    #    - CodeSearchNet / TheVault_Csharp => calc_codebleu=False
    #    - Other groups (e.g., CodeTrans, BFP) => calc_codebleu=True
    synthetic_examples = [
        {
            "dataset": "CodeSearchNet",
            "prediction": "returns the sum of two numbers",
            "reference": "return sum of two numbers",
        },
        {
            "dataset": "TheVault_Csharp",
            "prediction": "validates user input and throws exception",
            "reference": "validate user input and throw exception",
        },
        {
            "dataset": "CodeTrans",
            "prediction": "public static int square(int x) { return x * x; }",
            "reference": "public static int square(int n) { return n * n; }",
        },
        {
            "dataset": "BFP",
            "prediction": "for (int i = 0; i < arr.length; i++) { sum += arr[i]; }",
            "reference": "for (int i = 0; i < arr.length; i++) { total += arr[i]; }",
        },
    ]

    grouped_predictions = [e["prediction"] for e in synthetic_examples]
    grouped_references = [e["reference"] for e in synthetic_examples]
    grouped_datasets = [e["dataset"] for e in synthetic_examples]

    grouped_metrics = compute_grouped_metrics(
        grouped_predictions,
        grouped_references,
        grouped_datasets,
    )

    print("\n=== Grouped metrics by dataset (mixed CodeBLEU policy) ===")
    for k in sorted(grouped_metrics.keys()):
        print(f"{k}: {grouped_metrics[k]}")

    print("\n=== Dataset -> output language (from task_info) ===")
    for ds_name in sorted(set(grouped_datasets)):
        output_lang = DATASET_TO_OUTPUT_LANG.get(ds_name, "N/A")
        print(f"{ds_name}: {output_lang}")