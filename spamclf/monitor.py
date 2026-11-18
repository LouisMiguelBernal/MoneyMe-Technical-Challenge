import hashlib
import json
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from prometheus_client import Counter, Histogram

from .config import PREDICTION_LOG

PSI_DRIFT = 0.2
MIN_ROWS = 200  # below this PSI is noise

PREDICTIONS = Counter("spamclf_predictions_total", "Predictions served", ["classifier", "label"])
FAILURES = Counter("spamclf_failures_total", "Requests answered with 503", ["classifier"])
LATENCY = Histogram("spamclf_latency_seconds", "Inference time", ["classifier"],
                    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 120))
ML_SCORE = Histogram("spamclf_ml_score", "ML decision score", buckets=(-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3))
CHARS = Histogram("spamclf_message_chars", "Message length", buckets=(20, 50, 100, 160, 320, 640, 2000))

_lock = threading.Lock()


def record(classifier, text, label, latency_ms, model, score=None):
    PREDICTIONS.labels(classifier, label).inc()
    LATENCY.labels(classifier).observe(latency_ms / 1000)
    CHARS.observe(len(text))
    if score is not None:
        ML_SCORE.observe(score)
    if not PREDICTION_LOG:
        return
    # hash only, never the text
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "classifier": classifier, "model": model, "label": label, "score": score,
        "latency_ms": latency_ms, "chars": len(text),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    path = Path(PREDICTION_LOG)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def failure(classifier):
    FAILURES.labels(classifier).inc()


def bins(values, n=10):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)))
    return {"edges": edges.tolist(), "props": (np.histogram(values, bins=edges)[0] / len(values)).tolist()}


def psi(ref, values):
    edges = np.asarray(ref["edges"])
    cur = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)[0] / max(len(values), 1)
    exp, act = np.clip(ref["props"], 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def reference(scores, chars, labels):
    return {"ml_score": bins(scores), "chars": bins(chars), "spam_share": float(np.mean(labels == "spam"))}


def report(rows, ref):
    by = defaultdict(list)
    for r in rows:
        by[r["classifier"]].append(r)
    out = {"predictions": {k: len(v) for k, v in by.items()}, "reference_spam_share": ref["spam_share"]}
    for clf, rs in by.items():
        out[f"{clf}_spam_share"] = float(np.mean([r["label"] == "spam" for r in rs]))
    if by["ml"]:
        out["psi_ml_score"] = psi(ref["ml_score"], [r["score"] for r in by["ml"]])
        out["psi_chars"] = psi(ref["chars"], [r["chars"] for r in by["ml"]])
    # messages sent to both classifiers; needs no labels
    last = {clf: {r["text_sha256"]: r["label"] for r in rs} for clf, rs in by.items()}
    common = last.get("ml", {}).keys() & last.get("llm", {}).keys()
    if common:
        out["ml_llm_disagreement"] = float(np.mean([last["ml"][h] != last["llm"][h] for h in common]))
        out["messages_seen_by_both"] = len(common)
    enough = len(by["ml"]) >= MIN_ROWS
    out["drift"] = any(out[k] > PSI_DRIFT for k in ("psi_ml_score", "psi_chars")) if enough else None
    return out
