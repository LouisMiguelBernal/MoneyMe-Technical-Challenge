import os
import tempfile
from pathlib import Path

os.environ["LLM_WARMUP"] = "0"
os.environ["PREDICTION_LOG"] = str(Path(tempfile.mkdtemp()) / "predictions.jsonl")
