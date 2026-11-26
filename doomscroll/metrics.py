"""Classification metrics written out by hand (no sklearn needed), so the
numbers in reports are easy to read and check.

Predictions may be UNPARSED (the model's output had no usable label). That
counts as a miss for the true class but not as a false positive for any class.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Sequence

from doomscroll.schema import LABELS, UNPARSED


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> dict:
    """Precision, recall, F1 and support for each label."""
    out = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    """Unweighted mean of per-label F1 (each class counts equally)."""
    scores = per_class(y_true, y_pred, labels)
    return sum(s["f1"] for s in scores.values()) / len(labels)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> dict:
    """{true_label: {pred_label: count}}; columns include UNPARSED."""
    columns = list(labels) + [UNPARSED]
    counts = Counter(zip(y_true, y_pred))
    return {t: {p: counts[(t, p)] for p in columns} for t in labels}


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Resample the test set with replacement n_boot times, recompute the metric
    each time, and take the alpha/2 and 1-alpha/2 percentiles. A wide
    interval means the test set is too small to trust small differences.
    """
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(metric([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    scores.sort()
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two annotators, corrected for chance.

    1.0 = perfect, 0 = no better than chance. Above ~0.6 is usually
    considered substantial agreement.
    """
    if len(a) != len(b):
        raise ValueError("annotation lists must be the same length")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def summarize(y_true: Sequence[str], y_pred: Sequence[str], valid_json: Sequence[bool] | None = None,
              n_boot: int = 1000, seed: int = 0) -> dict:
    """Everything evaluate.py reports for one system."""
    result = {
        "n": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci95": bootstrap_ci(y_true, y_pred, accuracy, n_boot, seed=seed),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci95": bootstrap_ci(y_true, y_pred, macro_f1, n_boot, seed=seed),
        "per_class": per_class(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred),
        "unparsed_rate": sum(p == UNPARSED for p in y_pred) / len(y_pred) if y_pred else 0.0,
    }
    if valid_json is not None:
        result["valid_json_rate"] = sum(valid_json) / len(valid_json) if valid_json else 0.0
    return result
