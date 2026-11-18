FROM python:3.12-slim AS build
RUN python -m venv /venv
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
RUN useradd --create-home app
COPY --from=build /venv /venv
WORKDIR /app
COPY spamclf spamclf
COPY models models
RUN mkdir logs && chown app logs
USER app
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PREDICTION_LOG=/app/logs/predictions.jsonl
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=6 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "spamclf.api:app", "--host", "0.0.0.0", "--port", "8000"]
