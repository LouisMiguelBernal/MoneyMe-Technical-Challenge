import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from . import llm, ml, monitor
from .config import LLM_MODEL, LLM_WARMUP, MAX_CHARS

state = {"llm_warm": False}


def warm_llm():
    # load the model before the first request; no timeout, a cancelled load restarts from scratch
    client = httpx.Client(timeout=None)
    while True:
        try:
            llm.classify("warm-up", client=client, retries=0)
            state["llm_warm"] = True
            return
        except llm.LLMError:
            time.sleep(10)


@asynccontextmanager
async def lifespan(app):
    state["model"] = ml.load()
    if LLM_WARMUP:
        threading.Thread(target=warm_llm, daemon=True).start()
    yield


app = FastAPI(title="SMS Spam Classifier", version="1.0.0", lifespan=lifespan)


class SMS(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, examples=["Free entry! Text WIN to 80888"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class Result(BaseModel):
    label: Literal["spam", "ham"]
    classifier: Literal["ml", "llm"]
    model: str
    score: float | None = None
    latency_ms: float


def elapsed(t0):
    return round((time.perf_counter() - t0) * 1000, 2)


@app.post("/v1/classify/ml", response_model=Result)
def classify_ml(sms: SMS):
    t0 = time.perf_counter()
    model = state["model"]
    labels, scores = model.predict([sms.text])
    result = Result(label=labels[0], classifier="ml", model=model.card["model"],
                    score=float(scores[0]), latency_ms=elapsed(t0))
    monitor.record("ml", sms.text, result.label, result.latency_ms, result.model, result.score)
    return result


@app.post("/v1/classify/llm", response_model=Result, responses={503: {"description": "LLM backend unavailable"}})
def classify_llm(sms: SMS):
    t0 = time.perf_counter()
    try:
        label, _ = llm.classify(sms.text)
    except llm.LLMError as e:
        monitor.failure("llm")
        raise HTTPException(503, str(e)) from e
    result = Result(label=label, classifier="llm", model=LLM_MODEL, latency_ms=elapsed(t0))
    monitor.record("llm", sms.text, result.label, result.latency_ms, result.model)
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    card = state["model"].card
    return {"status": "ok", "model": card["model"], "model_sha256": card["sha256"],
            "threshold": card["threshold"]}


@app.get("/ready")
def ready():
    return {"ml": "model" in state, "llm": state["llm_warm"] and llm.ready(), "llm_model": LLM_MODEL}
