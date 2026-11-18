# SMS Spam Classifier

Two classifiers that label an SMS as `spam` or `ham`, built on the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and served with FastAPI.

| Classifier | How it works |
|---|---|
| ML | TF-IDF (word 1-2 grams + character 2-5 grams), LinearSVC |
| LLM | `gemma4:e4b` on Ollama. System prompt with a lender spam/scam policy, the 8 most similar training messages as examples, output restricted to `spam`/`ham` by a JSON schema |

## Run the API

Needs Docker with Compose. The LLM is a 6.6 GB model, so pick the option that fits the machine.

A. NVIDIA GPU (about 10 GB free disk):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

B. CPU only. Give Docker at least 12 GB of memory (Docker Desktop > Settings > Resources). Replies are much
slower; raise `LLM_TIMEOUT` if calls time out.

```bash
docker compose up --build
```

C. Ollama already running on the host (`ollama pull gemma4:e4b` first). Starts the API container only:

```bash
LLM_BASE_URL=http://host.docker.internal:11434 docker compose up --build --no-deps api
```

On Windows PowerShell set the variable first: `$env:LLM_BASE_URL="http://host.docker.internal:11434"`.

- API on http://localhost:8000, Swagger on `/docs`, endpoint reference in [docs/API.md](docs/API.md).
- The ML endpoint works as soon as the container is up; it does not depend on the LLM.
- With A or B the first start downloads the model, then the API warms it up. The LLM endpoint is ready
  when `GET /ready` returns `"llm": true`.

Call both endpoints:

```bash
pip install httpx
python examples/call_api.py
```

Optional settings (shell or `.env` next to `docker-compose.yml`):

| Variable | Default | |
|---|---|---|
| `API_PORT` | `8000` | host port |
| `LLM_MODEL` | `gemma4:e4b` | any Ollama model tag |
| `LLM_BASE_URL` | `http://ollama:11434` | use another Ollama server |
| `LLM_TIMEOUT` | `120` | seconds before the LLM endpoint returns 503 |

## Reproduce

Python 3.12+, and for the LLM steps a local [Ollama](https://ollama.com) with `gemma4:e4b`.

```bash
pip install -r requirements-dev.txt
python -m spamclf download     # raw UCI file
python -m spamclf stats
python -m spamclf train        # cross-validation, model + threshold -> models/
python -m spamclf eval-ml      # test split
python -m spamclf tune-llm     # prompt and examples, on a validation slice of train
python -m spamclf eval-llm     # test split
python -m spamclf probe        # data/probe_set.csv
python -m spamclf predict "Free entry! Text WIN to 80888" --clf ml
pytest
```

Each command writes to `reports/`. [sms_spam.ipynb](sms_spam.ipynb) walks through the data, the baseline,
the model comparison and both classifiers' results from those files.

## Results

Test split: 1,035 messages (131 spam), not used for training or for choosing the ML model and threshold.

| | Precision | Recall | F1 | Accuracy | Ham blocked |
|---|---|---|---|---|---|
| Always ham | 0.000 | 0.000 | 0.000 | 0.873 | 0.0% |
| Keyword rule | 0.674 | 0.885 | 0.766 | 0.931 | 6.2% |
| ML | 0.955 | 0.962 | 0.958 | 0.989 | 0.7% |
| LLM | 0.961 | 0.931 | 0.946 | 0.986 | 0.6% |

- F1 95% bootstrap CI: ML 0.931-0.981, LLM 0.913-0.973. McNemar on the paired predictions: p = 0.63,
  so the two are not distinguishable on this data.
- Latency p50: ML 4 ms on CPU, LLM 329 ms on an RTX 4050. No failed LLM calls; a repeat run of 100
  messages gave identical labels.
- ML model: LinearSVC had the best PR-AUC of five models in grouped CV. Logistic regression, the
  baseline, was within 2 messages.
- LLM: the lender policy prompt took ham blocked from 26 to 5 against a generic prompt (p < 0.001).
  Retrieved examples beat fixed ones on validation but not significantly on test. Every version tried
  is in [reports/llm_variants_test.json](reports/llm_variants_test.json).
- Probe set of 50 hand-written modern messages: ML 68%, LLM 98%. The ML model blocks most real
  transactional and lender notices; the LLM let all 20 through and caught all 10 loan scams.

## Method

- Raw file parsed with quoting off: 5,574 rows (4,827 ham / 747 spam). Default CSV parsing returns 5,572.
- 403 exact duplicates removed. Messages that differ only in digits or URLs form a template group, and
  no group is on both sides of the split.
- Threshold set on out-of-fold train scores so that at most 0.5% of ham is blocked.
- LLM prompt and retrieval chosen on train only. The served retrieval pool is the train split.
- Tests check row counts, the split, that shuffled labels score at chance, and that the API returns the
  same predictions as the offline model.

## Monitoring

- `GET /metrics`: Prometheus counters and histograms (predictions, LLM failures, latency, ML score,
  message length).
- Each prediction is logged to `logs/predictions.jsonl` with label, score, latency and length. The text
  is stored as a sha256 hash only.
- `docker compose exec api python -m spamclf monitor` compares the log with the test-set reference:
  spam share, PSI of score and length (above 0.2 is drift, null under 200 predictions) and ML/LLM
  disagreement, which needs no labels.

## Limits

- The LLM classifier runs on a local model (Gemma 4 E4B on Ollama), so its speed depends on the host
  machine. A GPU is recommended.

## Layout

```
spamclf/         config, data, ml, llm, evaluate, monitor, cli, api
models/          trained model, model card, LLM prompt choice, retrieval pool
reports/         generated metrics and predictions
sms_spam.ipynb   analysis notebook
examples/        call_api.py
tests/           pytest
docs/API.md      endpoint reference
```
