import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
MESSAGES = [
    "WINNER!! You have won a 1000 pound prize. Call 09061701461 to claim now",
    "Ok lar, see you at 7 then",
]

for path in ("/v1/classify/ml", "/v1/classify/llm"):
    for text in MESSAGES:
        r = httpx.post(BASE + path, json={"text": text}, timeout=300)
        print(path, r.status_code, r.json())
