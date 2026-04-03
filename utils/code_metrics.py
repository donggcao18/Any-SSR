from __future__ import annotations

import collections
import math
import re
import string
import xml.sax.saxutils
from typing import Dict, Iterable, List, Tuple


def normalize_text(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    s = (s or "").lower().replace("<pad>", "").replace("</s>", "")
    return white_space_fix(remove_articles(remove_punc(s)))


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def _get_ngrams(toks: List[str], max_order: int) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for o in range(1, max_order + 1):
        for i in range(0, len(toks) - o + 1):
            c[tuple(toks[i : i + o])] += 1
    return c


def bleu(refs: List[List[str]], hyps: List[str], max_order: int = 4, smooth: bool = False) -> float:
    """Simple corpus BLEU for tasks except CodeSearchNet/TheVault (user version).

    Args:
        refs: list over examples; each item is list of reference strings
        hyps: list of hypothesis strings
    """

    matches = [0] * max_order
    possibles = [0] * max_order
    ref_len = 0
    hyp_len = 0
    for rlist, hyp in zip(refs, hyps):
        r_tokens_list = [r.split() for r in rlist]
        h = hyp.split()
        if not r_tokens_list:
            continue
        ref_len += min(len(r) for r in r_tokens_list)
        hyp_len += len(h)
        merged: collections.Counter = collections.Counter()
        for r in r_tokens_list:
            merged |= _get_ngrams(r, max_order)
        h_counts = _get_ngrams(h, max_order)
        overlap = h_counts & merged
        for ng in overlap:
            matches[len(ng) - 1] += overlap[ng]
        for o in range(1, max_order + 1):
            p = len(h) - o + 1
            if p > 0:
                possibles[o - 1] += p

    prec = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            prec[i] = (matches[i] + 1.0) / (possibles[i] + 1.0)
        else:
            prec[i] = (matches[i] / possibles[i]) if possibles[i] > 0 else 0.0

    geo = math.exp(sum((1.0 / max_order) * math.log(p) for p in prec)) if min(prec) > 0 else 0.0
    ratio = float(hyp_len) / max(1, ref_len)
    bp = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / max(ratio, 1e-9))
    return geo * bp


# ---- Faithful Smooth BLEU (for CodeSearchNet/TheVault) ----

_normalize1_patterns = [
    (r"<skipped>", ""),
    (r"-\n", ""),
    (r"\n", " "),
]
_normalize1 = [(re.compile(pat), rep) for (pat, rep) in _normalize1_patterns]

_normalize2_patterns = [
    (r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 "),
    (r"([^0-9])([\.,])", r"\1 \2 "),
    (r"([\.,])([^0-9])", r" \1 \2"),
    (r"([0-9])(-)", r"\1 \2 "),
]
_normalize2 = [(re.compile(pat), rep) for (pat, rep) in _normalize2_patterns]


def _split_tokens_like_ref(s: str, *, preserve_case: bool = False, nonorm: bool = False) -> List[str]:
    if nonorm:
        return s.split()
    if not isinstance(s, str):
        s = " ".join(s)
    for (pattern, repl) in _normalize1:
        s = pattern.sub(repl, s)
    s = xml.sax.saxutils.unescape(s, {"&quot;": '"'})
    s = f" {s} "
    if not preserve_case:
        s = s.lower()
    for (pattern, repl) in _normalize2:
        s = pattern.sub(repl, s)
    return s.split()


def _count_ngrams(words: List[str], n: int = 4) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ng = tuple(words[i : i + k])
            counts[ng] = counts.get(ng, 0) + 1
    return counts


def _cook_refs(
    refs: List[str], *, n: int = 4, preserve_case: bool = False, nonorm: bool = False
) -> Tuple[List[int], Dict[Tuple[str, ...], int]]:
    refs_tok = [_split_tokens_like_ref(r, preserve_case=preserve_case, nonorm=nonorm) for r in refs]
    maxcounts: Dict[Tuple[str, ...], int] = {}
    for ref in refs_tok:
        counts = _count_ngrams(ref, n=n)
        for ng, c in counts.items():
            if c > maxcounts.get(ng, 0):
                maxcounts[ng] = c
    return [len(ref) for ref in refs_tok], maxcounts


def _cook_test(
    hyp: str,
    cooked_refs: Tuple[List[int], Dict[Tuple[str, ...], int]],
    *,
    n: int = 4,
    eff_ref_len: str = "shortest",
    preserve_case: bool = False,
    nonorm: bool = False,
) -> Dict[str, object]:
    reflens, refmaxcounts = cooked_refs
    hyp_tok = _split_tokens_like_ref(hyp, preserve_case=preserve_case, nonorm=nonorm)

    result: Dict[str, object] = {}
    testlen = len(hyp_tok)
    result["testlen"] = testlen

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / max(1, len(reflens))
    elif eff_ref_len == "closest":
        best_ref = reflens[0]
        min_diff = abs(reflens[0] - testlen)
        for rl in reflens[1:]:
            d = abs(rl - testlen)
            if d < min_diff:
                min_diff = d
                best_ref = rl
        result["reflen"] = best_ref
    else:
        raise ValueError("eff_ref_len must be one of {'shortest','average','closest'}")

    result["guess"] = [max(testlen - k + 1, 0) for k in range(1, n + 1)]
    result["correct"] = [0] * n

    h_counts = _count_ngrams(hyp_tok, n=n)
    for ng, c in h_counts.items():
        result["correct"][len(ng) - 1] += min(refmaxcounts.get(ng, 0), c)

    return result


def _score_cooked(allcomps: Iterable[Dict[str, object]], *, n: int = 4, smooth: int = 1) -> List[float]:
    total = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}
    for comps in allcomps:
        total["testlen"] += int(comps["testlen"])  # type: ignore[arg-type]
        total["reflen"] += int(comps["reflen"])  # type: ignore[arg-type]
        for k in range(n):
            total["guess"][k] += int(total["guess"][k] + int(comps["guess"][k])) - int(total["guess"][k])  # type: ignore[index]
            total["correct"][k] += int(total["correct"][k] + int(comps["correct"][k])) - int(total["correct"][k])  # type: ignore[index]

    logbleu = 0.0
    per_order_logs: List[float] = []
    for k in range(n):
        correct = total["correct"][k]
        guess = total["guess"][k]
        addsmooth = 1 if (smooth == 1 and k > 0) else 0
        logbleu += math.log(correct + addsmooth + 1e-16) - math.log(guess + addsmooth + 1e-16)
        if guess == 0:
            per_order_logs.append(-1e7)
        else:
            per_order_logs.append(math.log(correct + 1e-16) - math.log(guess + 1e-16))

    logbleu /= float(n)
    per_order_logs.insert(0, logbleu)

    brev_penalty = min(0.0, 1.0 - float(total["reflen"] + 1) / float(total["testlen"] + 1))
    out: List[float] = []
    for i, val in enumerate(per_order_logs):
        if i == 0:
            val += brev_penalty
        out.append(math.exp(val))
    return out


def smooth_bleu(
    refs: List[List[str]],
    hyps: List[str],
    *,
    n: int = 4,
    smooth: int = 1,
    eff_ref_len: str = "shortest",
    preserve_case: bool = False,
    nonorm: bool = False,
) -> float:
    assert len(refs) == len(hyps), "refs and hyps must have the same length"
    allc = []
    for rlist, hyp in zip(refs, hyps):
        cooked_refs = _cook_refs(rlist, n=n, preserve_case=preserve_case, nonorm=nonorm)
        cooked_test = _cook_test(
            hyp,
            cooked_refs,
            n=n,
            eff_ref_len=eff_ref_len,
            preserve_case=preserve_case,
            nonorm=nonorm,
        )
        allc.append(cooked_test)
    return _score_cooked(allc, n=n, smooth=smooth)[0]
