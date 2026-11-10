"""Thin client for the teacher / exam-writer / judge models (any OpenAI-compatible chat endpoint).

Three DIFFERENT model families are used on purpose:
  data teacher  -> writes the training rows
  exam writer   -> writes the exam (never sees the training rows)
  judge         -> scores answers blind
and none of them is the model under test, so a family-specific habit cannot flatter the result.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("B2W_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
API_KEY = os.environ.get("B2W_API_KEY", "")   # sent as a Bearer token when set
# One model name per role, from four different families. There are no defaults: set the variables for your endpoint.
ROLES = {
    "teacher": os.environ.get("B2W_TEACHER", ""),
    "exam": os.environ.get("B2W_EXAM", ""),
    "judge": os.environ.get("B2W_JUDGE", ""),
    "checker": os.environ.get("B2W_CHECKER", ""),   # confirms data rows; never sees the exam
}


def chat(role: str, prompt: str, *, max_tokens: int = 2500, retries: int = 6, temperature: float = 0.4) -> str:
    if not ROLES[role]:
        raise RuntimeError(f"set B2W_{role.upper()} to the model name for the '{role}' role")
    body = json.dumps({"model": ROLES[role], "max_tokens": max_tokens, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last = ""
    for attempt in range(retries):
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        req = urllib.request.Request(ENDPOINT, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                out = json.loads(r.read())
            text = out["choices"][0]["message"]["content"] or ""
            if text.strip():
                return text
            last = "empty answer"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(min(90, float(wait) if wait and wait.isdigit() else 8 * (attempt + 1)))
            continue
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            last = str(e)[:80]
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"{role} failed after {retries} tries: {last}")


def chat_json(role: str, prompt: str, **kw):
    """Ask for JSON and parse the first {...} or [...] block; one repair retry."""
    for _ in range(2):
        text = chat(role, prompt, **kw)
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        prompt += "\n\nReturn ONLY valid JSON, nothing else."
    raise ValueError(f"{role}: no valid JSON in answer")
