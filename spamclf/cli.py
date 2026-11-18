import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, llm, ml, monitor
from . import evaluate as ev
from .config import (
    FEWSHOT_PATH,
    LLM_MODEL,
    MAX_FPR,
    PREDICTION_LOG,
    PROBE_PATH,
    PROMPT_PATH,
    REFERENCE_PATH,
    REPORTS,
    RETRIEVAL_PATH,
    SEED,
)


def y01(df):
    return (df.label == "spam").astype(int).to_numpy()


def dump(name, obj):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj, indent=2))


def pct(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def stats(_):
    df = data.load()
    uniq = data.prepare(df)
    df["chars"], df["words"] = df.text.str.len(), df.text.str.split().str.len()
    length = df.groupby("label")[["chars", "words"]].agg(["mean", "median", "max"]).round(1)
    length.columns = ["_".join(c) for c in length.columns]
    dump("stats.json", {
        "sha256": data.sha256(data.RAW_PATH),
        "rows": len(df),
        "classes": df.label.value_counts().to_dict(),
        "spam_share": float((df.label == "spam").mean()),
        "exact_duplicates": int(df.duplicated("text").sum()),
        "unique_rows": len(uniq),
        "template_groups": int(uniq.group.nunique()),
        "length_by_class": length.to_dict("index"),
    })


def train(_):
    raw = data.load()
    tr, te = data.split(data.prepare(raw))
    X, y, g = tr.text.tolist(), y01(tr), tr.group.to_numpy()

    def cv(pipe, X=X, y=y, g=g):
        s = ml.oof_scores(pipe, X, y, g)
        thr = ml.pick_threshold(y, s)
        return ev.metrics(y, s >= thr, s) | {"threshold": thr}

    models = {name: cv(ml.build(name)) for name in ml.MODELS}
    best = max(models, key=lambda k: models[k]["pr_auc"])
    variants = {name: cv(ml.build(best, **kw)) for name, kw in ml.VARIANTS.items()}
    variant = max(variants, key=lambda k: variants[k]["pr_auc"])
    pipe = ml.build(best, **ml.VARIANTS[variant])

    # the usual leaky setup: duplicates kept, random split
    dup = raw[raw.text.isin(set(X))]
    shuffled = np.random.default_rng(SEED).permutation(y)
    report = {
        "models": models,
        "variants": variants,
        "chosen": {"model": best, "variant": variant},
        "leakage": {"random_split_with_duplicates": cv(pipe, dup.text.tolist(), y01(dup), None),
                    "group_split_deduplicated": variants[variant]},
        "label_shuffle": {"pr_auc": cv(pipe, y=shuffled)["pr_auc"], "chance": float(y.mean())},
    }
    ml.save(pipe.fit(X, y), variants[variant]["threshold"], {
        "model": best, "variant": variant, "max_fpr": MAX_FPR,
        "data_sha256": data.sha256(data.RAW_PATH), "n_train": len(tr), "n_test": len(te),
    })
    dump("cv.json", report)


def eval_ml(_):
    _, te = data.train_test()
    model, X, y = ml.load(), te.text.tolist(), y01(te)
    labels, s = model.predict(X)
    pred = (labels == "spam").astype(int)
    latency = []
    for t in X[:200]:
        t0 = time.perf_counter()
        model.predict([t])
        latency.append((time.perf_counter() - t0) * 1000)
    out = te.drop(columns="group").assign(ml_score=s, ml_pred=labels)
    path = REPORTS / "test_predictions.csv"
    if path.exists():  # keep llm_pred from the last eval-llm
        prev = pd.read_csv(path)
        if "llm_pred" in prev and prev.text.tolist() == out.text.tolist():
            out["llm_pred"] = prev.llm_pred
    out.to_csv(path, index=False)
    REFERENCE_PATH.write_text(json.dumps(monitor.reference(s, te.text.str.len().to_numpy(), labels), indent=2))
    dump("metrics_ml.json", {
        "n": len(te), "model": model.card["model"], "model_sha256": model.card["sha256"],
        "threshold": model.threshold,
        "test": ev.metrics(y, pred, s), "ci95": ev.bootstrap_ci(y, pred),
        "baselines": {"always_ham": ev.metrics(y, np.zeros_like(y)),
                      "keyword_rule": ev.metrics(y, ev.keyword_rule(X))},
        "latency_ms": pct(latency),
    })


def run_llm(texts, shots=None, system=None):
    # sequential on purpose: parallel calls get batched on the GPU and flip labels
    def one(text):
        t0 = time.perf_counter()
        try:
            pred, raw, error = *llm.classify(text, shots(text) if callable(shots) else shots, system=system), None
        except llm.LLMError as e:
            pred, raw, error = "error", None, str(e)
        return {"pred": pred, "raw": raw, "error": error,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    return [one(t) for t in texts]


def llm_pred(y, rows):
    # a failed call counts as wrong
    return np.array([{"spam": 1, "ham": 0}.get(r["pred"], 1 - t) for t, r in zip(y, rows)])


def tune_llm(_):
    tr, _ = data.train_test()
    shots = pd.concat([tr[tr.label == c].sample(4, random_state=SEED) for c in ("spam", "ham")])
    shots = shots.sample(frac=1, random_state=SEED)
    rest = tr.drop(shots.index)
    val = pd.concat([rest[rest.label == "spam"].sample(100, random_state=SEED),
                     rest[rest.label == "ham"].sample(200, random_state=SEED)])
    # pool without the validation messages or their templates
    pool = tr[~tr.group.isin(val.group)][["text", "label"]].to_dict("records")
    fixed = shots[["text", "label"]].to_dict("records")
    r8, r4 = llm.Retriever(pool, k=8), llm.Retriever(pool, k=8, per_class=True)
    variants = {
        "generic_zero_shot": ("generic", []),
        "generic_fixed_8": ("generic", fixed),
        "generic_retrieved_8": ("generic", r8),
        "generic_retrieved_4_per_class": ("generic", r4),
        "lender_policy_zero_shot": ("lender_policy", []),
        "lender_policy_retrieved_8": ("lender_policy", r8),
    }
    y = y01(val)
    results = {k: ev.metrics(y, llm_pred(y, run_llm(val.text.tolist(), shots, system)))
               for k, (system, shots) in variants.items()}
    chosen = max(results, key=lambda k: (results[k]["f1"], -results[k]["fp"]))
    system, picked = variants[chosen]
    PROMPT_PATH.write_text(json.dumps({"prompt": system}))
    FEWSHOT_PATH.write_text(json.dumps(fixed if picked else [], indent=2, ensure_ascii=False), encoding="utf-8")
    if isinstance(picked, llm.Retriever):
        # serve with the full train split
        full = llm.Retriever(tr[["text", "label"]].to_dict("records"), picked.k, picked.per_class)
        RETRIEVAL_PATH.write_text(json.dumps(full.config(), ensure_ascii=False), encoding="utf-8")
    else:
        RETRIEVAL_PATH.unlink(missing_ok=True)
    dump("llm_prompt_selection.json", {"n_validation": len(val), "results": results, "chosen": chosen})


def eval_llm(_):
    _, te = data.train_test()
    for fn in (llm.retriever, llm.fewshot, llm.prompt_name):
        fn.cache_clear()
    X, y = te.text.tolist(), y01(te)
    rows = run_llm(X)
    pred = llm_pred(y, rows)
    again = run_llm(X[:100])
    with open(REPORTS / "llm_raw.jsonl", "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"text": text, "label": label, **row}, ensure_ascii=False) + "\n"
            for text, label, row in zip(X, te.label, rows)
        )
    dump("metrics_llm.json", {
        "n": len(te), "model": LLM_MODEL, "prompt": llm.prompt_name(),
        "shots": f"{r.k} retrieved" if (r := llm.retriever()) else f"{len(llm.fewshot())} fixed",
        "test": ev.metrics(y, pred), "ci95": ev.bootstrap_ci(y, pred),
        "failed_calls": sum(r["pred"] == "error" for r in rows),
        "repeat_agreement": float(np.mean([a["pred"] == b["pred"] for a, b in zip(rows, again)])),
        "latency_ms": pct([r["latency_ms"] for r in again]),
    })
    both = pd.read_csv(REPORTS / "test_predictions.csv").assign(llm_pred=[r["pred"] for r in rows])
    both.to_csv(REPORTS / "test_predictions.csv", index=False)
    both[(both.ml_pred != both.label) | (both.llm_pred != both.label)].to_csv(
        REPORTS / "errors.csv", index=False
    )
    dump("comparison.json", {"first": "ml", "second": "llm",
                             **ev.mcnemar(y, (both.ml_pred == "spam").astype(int), pred)})


def probe(_):
    df = pd.read_csv(PROBE_PATH)
    texts = df.text.tolist()
    df["ml_pred"] = ml.load().predict(texts)[0]
    df["llm_pred"] = [r["pred"] for r in run_llm(texts)]
    df.to_csv(REPORTS / "probe_predictions.csv", index=False)
    dump("probe.json", {
        clf: {"overall": float((df[f"{clf}_pred"] == df.label).mean()),
              **(df[f"{clf}_pred"] == df.label).groupby(df.kind).mean().round(3).to_dict()}
        for clf in ("ml", "llm")
    } | {"n": len(df), "counts": df.kind.value_counts().to_dict()})


def drift(args):
    path = Path(args.log or PREDICTION_LOG)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["ts"] >= args.since] if args.since else rows
    print(json.dumps(monitor.report(rows, json.loads(REFERENCE_PATH.read_text())), indent=2))


def predict(args):
    if args.clf == "ml":
        labels, s = ml.load().predict([args.text])
        print(labels[0], round(float(s[0]), 4))
    else:
        print(llm.classify(args.text)[0])


def main():
    parser = argparse.ArgumentParser(prog="spamclf", description="SMS spam classifiers")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "download": (lambda _: print(data.download()), "fetch the raw UCI file"),
        "stats": (stats, "basic statistics of the data set"),
        "train": (train, "cross-validate the ML models, save the best with its threshold"),
        "eval-ml": (eval_ml, "score the ML model on the test split"),
        "tune-llm": (tune_llm, "pick the LLM prompt and examples on a validation slice of train"),
        "eval-llm": (eval_llm, "score the LLM on the test split and compare with ML"),
        "probe": (probe, "run both classifiers on data/probe_set.csv"),
    }
    for name, (fn, text) in commands.items():
        sub.add_parser(name, help=text).set_defaults(fn=fn)
    m = sub.add_parser("monitor", help="drift report from the API prediction log")
    m.add_argument("--log", help="prediction log path (default: PREDICTION_LOG)")
    m.add_argument("--since", help="only predictions at or after this ISO time, e.g. 2026-11-17")
    m.set_defaults(fn=drift)
    p = sub.add_parser("predict", help="classify one message")
    p.add_argument("text", help="the SMS text")
    p.add_argument("--clf", choices=["ml", "llm"], default="ml", help="classifier to use (default: ml)")
    p.set_defaults(fn=predict)
    args = parser.parse_args()
    args.fn(args)
