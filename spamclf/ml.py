import hashlib
import json
import re

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.svm import LinearSVC

from .config import CARD_PATH, MAX_FPR, MODEL_PATH, SEED


def xgboost():
    # imported here so the API image does not need xgboost
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
        scale_pos_weight=6.9, tree_method="hist", n_jobs=-1, random_state=SEED,  # ham:spam in train
    )


MODELS = {
    "complement_nb": lambda: ComplementNB(alpha=0.1),
    "logreg": lambda: LogisticRegression(C=10, class_weight="balanced", max_iter=2000),
    "linear_svc": lambda: LinearSVC(C=1, class_weight="balanced", random_state=SEED),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
    ),
    "xgboost": xgboost,
}
VARIANTS = {"raw": {}, "cleaned": {"cleaned": True}, "raw+hand": {"hand": True}}


def clean(text):
    words = re.sub(r"[^a-z ]+", " ", text.lower()).split()
    return " ".join(w for w in words if w not in ENGLISH_STOP_WORDS)


def hand_features(texts):
    return np.array([
        [
            len(t),
            sum(c.isdigit() for c in t),
            sum(c.isupper() for c in t) / max(len(t), 1),
            len(re.findall(r"[£$€]", t)),
            int(bool(re.search(r"https?://|www\.", t.lower()))),
        ]
        for t in texts
    ])


def build(name, cleaned=False, hand=False):
    pre = clean if cleaned else None
    parts = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=pre)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, preprocessor=pre)),
    ]
    if hand:
        parts.append(("hand", make_pipeline(FunctionTransformer(hand_features), MaxAbsScaler())))
    return make_pipeline(FeatureUnion(parts), MODELS[name]())


def scores(pipe, texts):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    return pipe.decision_function(texts)


def oof_scores(pipe, texts, y, groups=None):
    cv = (StratifiedGroupKFold if groups is not None else StratifiedKFold)(
        n_splits=5, shuffle=True, random_state=SEED
    )
    method = "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"
    out = cross_val_predict(pipe, texts, y, groups=groups, cv=cv, method=method)
    return out[:, 1] if out.ndim == 2 else out


def pick_threshold(y, s, max_fpr=MAX_FPR):
    fpr, _, thr = roc_curve(y, s, drop_intermediate=False)
    return float(thr[fpr <= max_fpr][-1])


class Model:
    def __init__(self, pipeline, threshold, card):
        self.pipeline, self.threshold, self.card = pipeline, threshold, card

    def predict(self, texts):
        s = scores(self.pipeline, texts)
        return np.where(s >= self.threshold, "spam", "ham"), s


def save(pipeline, threshold, card):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, MODEL_PATH)
    card |= {
        "threshold": threshold,
        "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "sklearn": sklearn.__version__,
    }
    CARD_PATH.write_text(json.dumps(card, indent=2))


def load():
    card = json.loads(CARD_PATH.read_text())
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != card["sha256"]:
        raise RuntimeError("model artifact does not match model_card.json")
    return Model(**joblib.load(MODEL_PATH), card=card)
