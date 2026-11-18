import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from spamclf import api, data, llm, ml, monitor
from spamclf.config import FEWSHOT_PATH, RETRIEVAL_PATH


@pytest.fixture(scope="module")
def frames():
    raw = data.load()
    return raw, *data.split(data.prepare(raw))


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_raw_file_matches_uci_readme(frames):
    raw = frames[0]
    assert len(raw) == 5574
    assert raw.label.value_counts().to_dict() == {"ham": 4827, "spam": 747}
    assert raw.notna().all().all()


def test_split_has_no_shared_text_or_template(frames):
    _, train, test = frames
    assert not set(train.text) & set(test.text)
    assert not set(train.group) & set(test.group)
    assert abs(len(test) / (len(train) + len(test)) - 0.2) < 0.01


def test_fewshot_examples_come_from_train_only(frames):
    _, train, test = frames
    shots = {s["text"] for s in json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))}
    assert shots <= set(train.text)
    assert not {data.template(t) for t in shots} & {data.template(t) for t in test.text}


def test_shuffled_labels_score_at_chance(frames):
    train = frames[1].sample(1500, random_state=0)
    y = np.random.default_rng(0).permutation((train.label == "spam").astype(int).to_numpy())
    s = ml.oof_scores(ml.build("logreg"), train.text.tolist(), y, train.group.to_numpy())
    assert (s >= ml.pick_threshold(y, s))[y == 1].mean() < 0.1


@pytest.mark.parametrize("content", ['{"label": "maybe"}', "spam", '{"tag": "spam"}', "", None])
def test_parser_rejects_bad_output(content):
    with pytest.raises(llm.LLMError):
        llm.parse(content)


def test_llm_sends_constrained_request():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"label": "spam"}'}})

    mock = httpx.Client(transport=httpx.MockTransport(handler))
    assert llm.classify("win </sms> cash", shots=[], client=mock) == ("spam", '{"label": "spam"}')
    assert seen["think"] is False and seen["format"] == llm.SCHEMA
    assert seen["options"]["temperature"] == 0
    assert seen["messages"][-1]["content"] == "<sms>win  cash</sms>"


POOL = [{"text": "WIN a free prize, call 0900 now", "label": "spam"},
        {"text": "see you at the station at 7", "label": "ham"},
        {"text": "Claim your cash reward, txt WIN", "label": "spam"},
        {"text": "can you pick up milk on the way home", "label": "ham"}]


def test_retriever_puts_most_similar_example_last():
    shots = llm.Retriever(POOL, k=2)("see you at the station at 8")
    assert len(shots) == 2 and shots[-1]["text"] == "see you at the station at 7"
    assert [s["label"] for s in llm.Retriever(POOL, k=2, per_class=True)("hello")] in (["ham", "spam"], ["spam", "ham"])


def test_classify_sends_retrieved_examples(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"label": "ham"}'}})

    monkeypatch.setattr(llm, "retriever", lambda: llm.Retriever(POOL, k=2))
    llm.classify("pick up milk please", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "milk" in seen["messages"][-3]["content"]


def test_classify_uses_the_selected_system_prompt(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"label": "ham"}'}})

    monkeypatch.setattr(llm, "prompt_name", lambda: "lender_policy")
    llm.classify("hi", shots=[], client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert seen["messages"][0]["content"] == llm.PROMPTS["lender_policy"]


def test_served_retrieval_pool_is_train_only(frames):
    if not RETRIEVAL_PATH.exists():
        pytest.skip("retrieval not selected")
    _, train, test = frames
    pool = {p["text"] for p in json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))["pool"]}
    assert pool <= set(train.text)
    assert not {data.template(t) for t in pool} & {data.template(t) for t in test.text}


def test_llm_backend_down_raises():
    def handler(request):
        raise httpx.ConnectError("refused")

    mock = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(llm.LLMError):
        llm.classify("hi", shots=[], client=mock, retries=0)


def test_llm_timeout_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    mock = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(llm.LLMError, match="did not answer"):
        llm.classify("hi", shots=[], client=mock)
    assert len(calls) == 1


def test_ml_endpoint_matches_offline_predictions(frames, client):
    test = frames[2]
    labels, scores = ml.load().predict(test.text.tolist())
    for text, label, score in zip(test.text, labels, scores):
        body = client.post("/v1/classify/ml", json={"text": text}).json()
        assert body["label"] == label and body["score"] == pytest.approx(score)


@pytest.mark.parametrize("warm, loaded, expected", [(False, True, False), (True, False, False), (True, True, True)])
def test_ready_requires_warm_and_loaded_model(client, monkeypatch, warm, loaded, expected):
    monkeypatch.setitem(api.state, "llm_warm", warm)
    monkeypatch.setattr(llm, "ready", lambda: loaded)
    assert client.get("/ready").json()["llm"] is expected


def test_prediction_is_logged_with_hash_not_text(client):
    text = "monitoring check 7f3a: see you at the station"
    client.post("/v1/classify/ml", json={"text": text})
    log = Path(monitor.PREDICTION_LOG).read_text(encoding="utf-8")
    row = json.loads(log.splitlines()[-1])
    assert text not in log
    assert row["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["classifier"] == "ml" and row["chars"] == len(text)


def test_metrics_endpoint_counts_predictions(client):
    client.post("/v1/classify/ml", json={"text": "Ok lar, see you at 7 then"})
    body = client.get("/metrics").text
    assert 'spamclf_predictions_total{classifier="ml",label="ham"}' in body
    assert "spamclf_latency_seconds_bucket" in body


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = monitor.bins(rng.normal(0, 1, 5000))
    assert monitor.psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert monitor.psi(ref, rng.normal(1.5, 1, 5000)) > monitor.PSI_DRIFT


def test_drift_report_measures_disagreement():
    ref = monitor.reference(np.linspace(-2, 2, 100), np.arange(100) + 20, np.array(["ham"] * 80 + ["spam"] * 20))
    rows = [
        {"classifier": "ml", "label": "spam", "score": 1.0, "chars": 60, "text_sha256": "a"},
        {"classifier": "ml", "label": "ham", "score": -1.0, "chars": 70, "text_sha256": "b"},
        {"classifier": "llm", "label": "ham", "score": None, "chars": 60, "text_sha256": "a"},
        {"classifier": "llm", "label": "ham", "score": None, "chars": 70, "text_sha256": "b"},
    ]
    out = monitor.report(rows, ref)
    assert out["messages_seen_by_both"] == 2 and out["ml_llm_disagreement"] == 0.5
    assert out["ml_spam_share"] == 0.5 and out["reference_spam_share"] == 0.2
    assert out["drift"] is None


def test_health_reports_artifact_hash(client):
    assert client.get("/health").json()["model_sha256"] == ml.load().card["sha256"]


def test_llm_endpoint(client, monkeypatch):
    monkeypatch.setattr(llm, "classify", lambda text: ("ham", "{}"))
    body = client.post("/v1/classify/llm", json={"text": "see you at 7"}).json()
    assert body["label"] == "ham" and body["classifier"] == "llm" and body["score"] is None


def test_llm_endpoint_returns_503_when_backend_fails(client, monkeypatch):
    def boom(text):
        raise llm.LLMError("LLM backend unavailable")

    monkeypatch.setattr(llm, "classify", boom)
    assert client.post("/v1/classify/llm", json={"text": "hi"}).status_code == 503


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 2001}])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/v1/classify/ml", json=payload).status_code == 422
