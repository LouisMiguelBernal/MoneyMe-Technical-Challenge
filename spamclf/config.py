import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_PATH = ROOT / "data" / "raw" / "SMSSpamCollection"
PROBE_PATH = ROOT / "data" / "probe_set.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "spam_model.joblib"))
CARD_PATH = MODEL_PATH.with_name("model_card.json")
FEWSHOT_PATH = MODEL_PATH.with_name("fewshot.json")
PROMPT_PATH = MODEL_PATH.with_name("prompt.json")
RETRIEVAL_PATH = MODEL_PATH.with_name("retrieval.json")
REFERENCE_PATH = MODEL_PATH.with_name("reference.json")
PREDICTION_LOG = os.getenv("PREDICTION_LOG", str(ROOT / "logs" / "predictions.jsonl"))
REPORTS = ROOT / "reports"

SEED = 42
MAX_FPR = 0.005  # max share of ham blocked
MAX_CHARS = 2000

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_WARMUP = os.getenv("LLM_WARMUP", "1") == "1"
