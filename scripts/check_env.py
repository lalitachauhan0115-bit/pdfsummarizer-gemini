#!/usr/bin/env python
"""Phase 0 gate: prove the local environment and Gemini credentials work.

Checks, in order:
  1. .env exists next to the project root and holds both variables
  2. the API key is accepted by the REST API (list models)
  3. the configured model is actually available to this key
  4. a live generateContent call succeeds (proves quota + model access)
  5. the LangChain path works (ChatGoogleGenerativeAI + a real invoke)

Run:  python scripts/check_env.py
Exit code 0 means every check passed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://generativelanguage.googleapis.com/v1beta"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


RETRYABLE = {429, 500, 502, 503, 504}


def http_json(url: str, payload: dict | None = None, attempts: int = 4) -> dict:
    """GET/POST JSON, retrying transient capacity errors with backoff."""
    data = json.dumps(payload).encode() if payload is not None else None
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            last = RuntimeError(f"HTTP {e.code}: {body[:400]}")
            if e.code not in RETRYABLE or attempt == attempts:
                raise last from e
            wait = 2 ** attempt
            print(f"  ... HTTP {e.code} on attempt {attempt}/{attempts}, retrying in {wait}s")
            time.sleep(wait)
    raise last  # pragma: no cover - loop always returns or raises


def main() -> int:
    # --- 1. .env ---------------------------------------------------------
    env_path = ROOT / ".env"
    if not env_path.exists():
        record(".env present", False, f"missing at {env_path}")
        print_report()
        return 1
    record(".env present", True, str(env_path))

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        record("python-dotenv importable", True)
    except ImportError as e:
        record("python-dotenv importable", False, str(e))
        print_report()
        return 1

    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    record("GEMINI_API_KEY set", bool(key), f"{len(key)} chars, starts {key[:4]!r}" if key else "empty")
    record("GEMINI_MODEL set", bool(model), model)
    if not key or not model:
        print_report()
        return 1

    # --- 2. key accepted -------------------------------------------------
    try:
        listing = http_json(f"{API}/models?key={key}&pageSize=200")
        names = [
            m["name"].removeprefix("models/")
            for m in listing.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        record("API key accepted", True, f"{len(names)} models support generateContent")
    except Exception as e:  # noqa: BLE001
        record("API key accepted", False, str(e))
        print_report()
        return 1

    # --- 3. model available ---------------------------------------------
    record(
        f"model '{model}' available",
        model in names,
        "" if model in names else "not in this key's model list",
    )
    if model not in names:
        print_report()
        return 1

    # --- 4. live generateContent -----------------------------------------
    started = time.time()
    try:
        resp = http_json(
            f"{API}/models/{model}:generateContent?key={key}",
            {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]},
        )
        text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        elapsed = time.time() - started
        record("live generateContent", True, f"replied {text!r} in {elapsed:.1f}s")
    except Exception as e:  # noqa: BLE001
        record("live generateContent", False, str(e)[:300])

    # --- 5. LangChain path ------------------------------------------------
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, temperature=0.0, google_api_key=key)
        msg = llm.invoke("Reply with exactly: LANGCHAIN_OK")
        record("langchain_google_genai invoke", True, f"replied {str(msg.text).strip()[:40]!r}")
    except Exception as e:  # noqa: BLE001
        record("langchain_google_genai invoke", False, f"{type(e).__name__}: {str(e)[:300]}")

    return print_report()


def print_report() -> int:
    width = max(len(r[0]) for r in results) + 2
    print("\n" + "=" * (width + 50))
    print("PHASE 0 ENVIRONMENT CHECK")
    print("=" * (width + 50))
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name.ljust(width)} {detail}")
    failed = [r for r in results if not r[1]]
    print("=" * (width + 50))
    if failed:
        print(f"{len(failed)} check(s) failed.\n")
        return 1
    print("All checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
