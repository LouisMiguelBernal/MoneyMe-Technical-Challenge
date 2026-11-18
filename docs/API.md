# API

Base URL: `http://localhost:8000` · interactive docs: `/docs` · OpenAPI schema: `/openapi.json`

## POST /v1/classify/ml
TF-IDF + LinearSVC, runs inside the API container.

## POST /v1/classify/llm
`gemma4:e4b` on Ollama, with the lender policy prompt and 8 retrieved examples.

Both endpoints take and return the same shape.

**Request**
```json
{"text": "WINNER!! You have won a 1000 pound prize. Call 09061701461 to claim now"}
```
`text`: string, 1–2000 characters, not blank.

**Response 200**
```json
{"label": "spam", "classifier": "ml", "model": "linear_svc", "score": 1.42, "latency_ms": 4.1}
```
| Field | Meaning |
|---|---|
| `label` | `spam` or `ham` |
| `classifier` | `ml` or `llm` |
| `model` | model that produced the label |
| `score` | ML only: decision score, `spam` when it is at or above the threshold reported by `/health`. `null` for the LLM |
| `latency_ms` | server-side time for this call |

**Errors**
| Status | When |
|---|---|
| 422 | `text` missing, blank or longer than 2000 characters |
| 503 | LLM endpoint only: Ollama unreachable, model not yet downloaded, no answer within `LLM_TIMEOUT` seconds, or the model returned something other than `spam`/`ham`. The body's `detail` gives the reason. No default label is ever returned |

## GET /health
```json
{"status": "ok", "model": "linear_svc", "model_sha256": "…", "threshold": -0.298}
```

## GET /ready
```json
{"ml": true, "llm": true, "llm_model": "gemma4:e4b"}
```
`llm` is `true` once a warm-up request sent by the API at start-up has succeeded and the model is
loaded in memory. With an Ollama outside Docker that unloads idle models, it reads `false` between
calls; the next call then reloads the model first.

## GET /metrics
Prometheus text format: `spamclf_predictions_total{classifier,label}`,
`spamclf_failures_total{classifier}`, `spamclf_latency_seconds{classifier}`, `spamclf_ml_score`,
`spamclf_message_chars`.

## Examples
```bash
curl -X POST http://localhost:8000/v1/classify/ml -H "Content-Type: application/json" -d '{"text": "Ok lar, see you at 7 then"}'
```
```bash
curl -X POST http://localhost:8000/v1/classify/llm -H "Content-Type: application/json" -d '{"text": "Ok lar, see you at 7 then"}'
```
```python
import httpx

r = httpx.post("http://localhost:8000/v1/classify/llm", json={"text": "Ok lar, see you at 7 then"}, timeout=300)
print(r.json())
```
`python examples/call_api.py` calls both endpoints and prints the responses.
