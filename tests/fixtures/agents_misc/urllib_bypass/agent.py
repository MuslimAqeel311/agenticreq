"""Non-httpx client (urllib) — outside S2. The hook cannot see it; the socket guard must count it as a miss."""

import json
import os
import urllib.request

req = urllib.request.Request(os.environ["OPENAI_BASE_URL"] + "/chat/completions",
                             data=json.dumps({"model": "gpt-4o", "messages": []}).encode(),
                             headers={"content-type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=2)
except Exception as e:
    print("failed as expected:", type(e).__name__)
