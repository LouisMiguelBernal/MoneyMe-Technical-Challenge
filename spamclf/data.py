import csv
import hashlib
import io
import re
import zipfile

import httpx
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import DATA_URL, RAW_PATH, SEED


def download():
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        body = httpx.get(DATA_URL, follow_redirects=True, timeout=60).raise_for_status().content
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            for name in ("SMSSpamCollection", "readme"):
                (RAW_PATH.parent / name).write_bytes(z.read(name))
    return RAW_PATH


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    # QUOTE_NONE: messages contain bare quotes that would otherwise merge rows
    return pd.read_csv(
        download(), sep="\t", header=None, names=["label", "text"],
        quoting=csv.QUOTE_NONE, encoding="utf-8", dtype=str,
    )


def template(text):
    t = re.sub(r"https?://\S+|www\.\S+", " url ", text.lower())
    t = re.sub(r"\d+", "0", t)
    return " ".join(re.sub(r"[^a-z0£$]+", " ", t).split())


def prepare(df):
    df = df.drop_duplicates("text").reset_index(drop=True)
    df["group"] = [hashlib.md5(template(t).encode()).hexdigest()[:12] for t in df["text"]]
    return df


def split(df):
    # one fold of five = 20% test, no template group on both sides
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(folds.split(df["text"], df["label"], df["group"]))
    return df.iloc[train].reset_index(drop=True), df.iloc[test].reset_index(drop=True)


def train_test():
    return split(prepare(load()))
