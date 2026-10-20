"""
generate_dataset/check_api.py
==============================
Quick sanity check for the configured model backend (local vLLM or remote API).

Run from master_thesis/ root:
    conda run -n master python -m generate_dataset.check_api

What it checks:
  1. API key is present (for api backend)
  2. GET /models — endpoint reachable + target model listed
  3. POST /chat/completions — real inference call + usage field returned
"""

import os
import sys
from pathlib import Path

# Load .env from mimo repo if key not already in environment
_env_path = Path(__file__).resolve().parents[2] / "mimo-claude-code-traces-1k" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import requests  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


def check_api():
    backend  = cfg.MODEL_BACKEND
    base_url = cfg.resolved_base_url()
    model    = cfg.resolved_model()
    api_key  = cfg.resolved_api_key()

    print("=" * 55)
    print(f"  Backend  : {backend.upper()}")
    print(f"  Endpoint : {base_url}")
    print(f"  Model    : {model}")
    print(f"  API key  : {'SET ✅' if api_key and api_key != 'not-needed' else 'NOT SET ❌'}")
    print("=" * 55)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    # ------------------------------------------------------------------
    # 1. Key check (api backend only)
    # ------------------------------------------------------------------
    if backend == "api" and (not api_key or api_key == "not-needed"):
        print(f"\n❌  FAIL: API key missing.")
        print(f"   Set {cfg.API_KEY_ENV} in your environment or .env file.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. GET /models
    # ------------------------------------------------------------------
    print("\n[1/2] GET /models ...")
    try:
        r = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        print(f"      Status    : {r.status_code} OK")
        print(f"      Available : {ids}")
        if model in ids:
            print(f"      Target    : '{model}' FOUND ✅")
        else:
            print(f"      Target    : '{model}' NOT FOUND ❌")
            print(f"      Available models listed above — update API_MODEL in config.py")
            sys.exit(1)
    except Exception as exc:
        print(f"      FAILED: {exc}")
        if backend == "local":
            print("      Is vLLM running?  python -m vllm.entrypoints.openai.api_server ...")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. POST /chat/completions
    # ------------------------------------------------------------------
    print("\n[2/2] POST /chat/completions ...")
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": "Reply with exactly one word: hello"}],
        "temperature": 0,
        "max_tokens":  16,
    }
    try:
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        print(f"      Status    : {r.status_code}")
        if r.status_code == 200:
            rj      = r.json()
            msg     = rj["choices"][0]["message"]
            content = (
                msg.get("content")
                or msg.get("reasoning")
                or msg.get("reasoning_content")
                or ""
            ).strip()
            usage   = rj.get("usage", {})
            print(f"      Response  : {content!r}")
            print(f"      usage     : {usage}")
            if usage:
                print(f"      usage populated ✅  (patch will capture this)")
            else:
                print(f"      usage empty ⚠️   (token metrics won't be captured)")
            print()
            print("API SANITY CHECK: PASSED ✅")
        else:
            print(f"      Response  : {r.text[:400]}")
            print()
            print("API SANITY CHECK: FAILED ❌")
            sys.exit(1)
    except Exception as exc:
        print(f"      FAILED: {exc}")
        print()
        print("API SANITY CHECK: FAILED ❌")
        sys.exit(1)


if __name__ == "__main__":
    check_api()
