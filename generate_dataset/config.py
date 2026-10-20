"""
generate_dataset/config.py
==========================
Single source of truth for every tunable setting in the trace-generation
pipeline.  Change values here; do not scatter magic numbers elsewhere.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUICK SWITCH — choose your model backend:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODEL_BACKEND = "local"   → vLLM running on this machine / cluster
  MODEL_BACKEND = "api"     → remote OpenAI-compatible API (Morpheus, OpenAI, etc.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ★  BACKEND SWITCH  ★
#    "local"  →  vLLM served on this machine / cluster node
#    "api"    →  remote OpenAI-compatible endpoint (Morpheus / OpenAI / etc.)
# ---------------------------------------------------------------------------
MODEL_BACKEND: str = "api"   # ← change this one line to swap backend

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parents[1]  # master_thesis/
OUTPUT_DIR = REPO_ROOT / "outputs" / "traces"     # where runs.jsonl + overall.json land

# ---------------------------------------------------------------------------
# LOCAL backend — vLLM running on localhost (or cluster node)
# ---------------------------------------------------------------------------
# Start vLLM with:
#   python -m vllm.entrypoints.openai.api_server \
#       --model google/gemma-4-12B-it --tensor-parallel-size 2 --port 8000

LOCAL_BASE_URL   = "http://localhost:8000/v1"
LOCAL_MODEL      = "google/gemma-4-12B-it"   # must match --model passed to vLLM
LOCAL_API_KEY    = "not-needed"              # vLLM ignores Authorization header

# ---------------------------------------------------------------------------
# API backend — remote OpenAI-compatible endpoint
# Follows the same pattern as LLMJudge._call_api in reward.py:
#   POST {API_BASE_URL}/chat/completions
#   Authorization: Bearer {API_KEY}
# ---------------------------------------------------------------------------
# Set API_KEY here OR export it as the env var named in API_KEY_ENV.
# The env var always wins over the hardcoded value (safer for git).

API_BASE_URL = "https://morpheus.cit.tum.de/api/v1"  # ← your API endpoint
API_MODEL    = "google/gemma-4-31B-it"                # ← model served at that endpoint
API_KEY_ENV  = "MORPHEUS_API_KEY"                     # env var that holds the key
API_KEY      = os.environ.get(API_KEY_ENV, "")        # populated from environment

# ---------------------------------------------------------------------------
# Shared inference settings (apply to both backends)
# ---------------------------------------------------------------------------
MAX_TOKENS   = 1024
TEMPERATURE  = 0.0
TIMEOUT_S    = 120        # per-request timeout in seconds

# ---------------------------------------------------------------------------
# AgentBench assignment
# ---------------------------------------------------------------------------
AGENT_NAME   = "gemma-local"   # logical name used inside AgentBench configs
TASK_NAME    = "os-std"        # change to "dbbench-std" to run DB tasks
CONCURRENCY  = 4               # parallel workers per task/agent

# ---------------------------------------------------------------------------
# Efficiency metrics — vLLM/API usage dict → friendly names in trace output
# ---------------------------------------------------------------------------
USAGE_KEY_MAP = {
    "prompt_tokens":     "input_tokens",
    "completion_tokens": "output_tokens",
    "total_tokens":      "total_tokens",
}

# ---------------------------------------------------------------------------
# Trace output
# ---------------------------------------------------------------------------
TRACES_OUTPUT_FILE = OUTPUT_DIR / "traces.jsonl"


# ---------------------------------------------------------------------------
# Resolved settings (read-only — derived from the switch above)
# ---------------------------------------------------------------------------
# Use these in run.py instead of branching on MODEL_BACKEND everywhere.

def resolved_base_url() -> str:
    return LOCAL_BASE_URL if MODEL_BACKEND == "local" else API_BASE_URL

def resolved_model() -> str:
    return LOCAL_MODEL if MODEL_BACKEND == "local" else API_MODEL

def resolved_api_key() -> str:
    if MODEL_BACKEND == "local":
        return LOCAL_API_KEY
    # Re-read the env at call time so a .env loaded after this module was
    # imported (e.g. by run.py) is still picked up.
    return os.environ.get(API_KEY_ENV, "") or API_KEY
