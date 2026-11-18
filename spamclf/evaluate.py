import re

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support

from .config import SEED

KEYWORDS = re.compile(
    r"\b(free|win|won|prize|claim|txt|text|urgent|cash|call now|reply|stop)\b|\d{5,}", re.IGNORECASE
)


def keyword_rule(texts):
    return np.array([int(bool(KEYWORDS.search(t))) for t in texts])


def metrics(y, pred, score=None):
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "accuracy": float((tp + tn) / len(y)), "ham_blocked_rate": float(fp / max(fp + tn, 1)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if score is not None:
        out["pr_auc"] = float(average_precision_score(y, score))
    return out


def bootstrap_ci(y, pred, n=2000):
    rng = np.random.default_rng(SEED)
    y, pred = np.asarray(y), np.asarray(pred)
    draws = [metrics(y[i], pred[i]) for i in rng.integers(0, len(y), (n, len(y)))]
    return {
        k: [float(v) for v in np.percentile([d[k] for d in draws], [2.5, 97.5])]
        for k in ("precision", "recall", "f1", "accuracy")
    }


def mcnemar(y, a, b):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": float(p)}
