# generate_dataset — Progress & Reproduction Guide

**Goal:** run the AgentBench FC `os-std` task with a language model served over an
**OpenAI-compatible API** (TUM Morpheus / Gemma 4), capture **token-efficiency
metrics** per sample, and **save the resulting traces locally** in a schema that
`mimo-claude-code-traces-1k/reward.py` can score directly.

**Status:** ✅ **COMPLETE** — 142 traces generated (144 samples, 2 errors), scored
with LLM judge. Mean reward: 0.5985. All outputs in `outputs/traces/` and 
`reward_scores/gemma-4-31B-it/`.

Last updated: 2026-10-15

---

## 1. Architecture (as actually built, not as the README describes)

```
         HOST (your machine / cluster login node)
  ┌──────────────────────────────────────────────┐
  │  run.py                                      │
  │   ├─ HTTPAgent ──────── HTTPS ──────────────►│──► Morpheus API
  │   │   (+ usage capture patch)                │     google/gemma-4-31B-it
  │   └─ TaskClient ───────┐                     │
  └────────────────────────┼─────────────────────┘
                           │ HTTP :5020/api
         DOCKER            ▼
  ┌──────────────────────────────────────────────┐
  │  agentrl-controller      (:5020)             │
  │        ▲                                     │
  │        │ registers                           │
  │  os-worker  ──spawns──►  local-os/default    │  ← agent's bash runs here
  │        │                 (sandbox container) │
  │        └──► redis:7  (sandbox locking/state) │
  └──────────────────────────────────────────────┘
```

Key facts discovered the hard way:

| Claim in README/docs | Reality in this repo |
|---|---|
| `python -m src.start_task` starts the server | **Does not exist.** FC version uses Docker Compose. |
| Controller is on port **5000** | Controller is on port **5020** only (both workers *and* clients). |
| `src/server/task_controller.py` exists | Does not exist; controller is the `jingbh/agentrl-controller` image. |
| `docs/Entrance_en.md` is current | It documents **AgentBench v0.2** and is stale for the FC version. |

`agentrl` is **not** installed on the host — it lives only inside the worker
image. Anything importing `agentrl` must run in the container.

---

## 2. One-time machine setup

### 2.1 macOS (what this was developed on)

Docker Desktop is *not* required; Colima is a lighter daemon.

```bash
brew install docker colima docker-compose docker-buildx
```

Register the Homebrew CLI plugins so `docker compose` / `docker buildx` are found —
add `cliPluginsExtraDirs` to `~/.docker/config.json`:

```json
{
  "auths": {},
  "currentContext": "colima",
  "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]
}
```

Start the daemon (first run downloads a ~700 MB VM image):

```bash
colima start --cpu 2 --memory 4 --disk 20
colima status          # verify
docker info            # verify
```

> Colima must be restarted after a reboot: `colima start`.

### 2.2 Linux / cluster

```bash
sudo apt-get install -y docker.io docker-compose-plugin docker-buildx-plugin
sudo systemctl start docker
```

### 2.3 Python environment

Host side only needs the *client* dependencies (no `torch`, no `agentrl`):

```bash
conda create -n master python=3.13 -y
conda activate master
pip install -r requirements.txt   # repo root
pip install fschat                # imported by src/client/agents (now optional)
```

Verified working with: `pydantic 2.13.4`, `requests 2.33.1`, `PyYAML 6.0.3`,
`tqdm 4.68.3`.

> The repo's `requirements.txt` pins old versions (`pydantic~=1.10`,
> `numpy~=1.23`) and recommends Python 3.9. Python 3.13 + pydantic 2 works for
> the client path after the fixes in §4, but is not what upstream tested.

### 2.4 API key

The Morpheus key is read from a `.env` file. Both locations are checked
automatically (first match wins, existing env vars take precedence):

1. `master_thesis/.env`
2. `../mimo-claude-code-traces-1k/.env`  ← where it currently lives

```
MORPHEUS_API_KEY=<your key>
```

Or export it: `export MORPHEUS_API_KEY=...`

---

## 3. Files in this folder

| File | Purpose | Status |
|---|---|---|
| `config.py` | Single source of truth. **`MODEL_BACKEND = "local" \| "api"`** switch, endpoints, model names, concurrency, output paths. | ✅ |
| `check_api.py` | Step 4 — verifies the model endpoint: key present, `GET /models`, real `POST /chat/completions`, confirms `usage` is populated. | ✅ passes |
| `setup_docker.py` | Step 5 — builds/pulls all 7 required images. Idempotent (`--skip-existing`), `--dry-run`, auto-enables BuildKit, falls back off the `docker.1ms.run` mirror. | ✅ passes |
| `docker-compose.local.yml` | Minimal local stack: controller + redis + os worker. (The full 5-task + freebase stack is `extra/docker-compose.yml`, for the cluster.) | ✅ |
| `serve_local.py` | Step 6 — `up` / `down` / `status` / `logs` for the local stack; waits for worker registration. | ✅ passes |
| `smoke_test.py` | Runs **one** `os-std` sample end-to-end to verify the pipeline. | ✅ verified |
| `agent_loop.py` | **Step 7a** — purpose-built FC controller client. `run_sample()` drives `start_sample` → model → `interact` loop. Returns full conversation + token usage. | ✅ complete |
| `run.py` | **Step 7b** — orchestrates parallel trace generation + post-processing to `traces.jsonl`. | ✅ complete |
| `configs/local-gemma.yaml` | `HTTPAgent` agent definition (url/headers/body/prompter). Not used in final implementation — `agent_loop.py` has its own `ModelClient`. | 🗂️ legacy |
| `configs/assignment.yaml` | Assignment skeleton from the legacy client path. Not used in final implementation. | 🗂️ legacy |

---

## 4. Changes made to the repo outside this folder

Five files. All are genuine bug fixes needed to get this far — **keep them when
moving to another machine** (they are committed in the working tree, see
`git status`).

| File | Change | Why |
|---|---|---|
| `src/client/agents/__init__.py` | `FastChatAgent` import wrapped in `try/except ImportError` | It eagerly imported `fastchat` → `torch`, breaking *all* `HTTPAgent` use. We never use FastChat. |
| `src/client/task.py` | `get_concurrency()` accepts `"ALIVE"` **string** as well as the `WorkerStatus` IntEnum, and skips `locked` workers | The FC controller returns `"status": "ALIVE"` (string); the old `IntEnum` compare (`ALIVE == 0`) was always `False`, so concurrency was **0** and the assigner silently dispatched nothing. |
| `src/server/tasks/os_interaction/Dockerfile` | pin `aiodocker==0.24.0` after `agentrl-worker` | `aiodocker 0.27` + current `aiohttp` raised `AttributeError: 'int' object has no attribute 'connect'` the moment the worker exec'd a command in a sandbox container. Hard blocker; this fixes it. |
| `src/server/tasks/os_interaction/task.py` | `b'\x1b\[\?2004[hl]'` → `rb'...'` | `SyntaxWarning: invalid escape sequence` on Python 3.12+. |
| `.gitignore` | ignore generated artifacts | housekeeping |

Also fixed inside `generate_dataset/configs/assignment.yaml`:
- controller port `5000` → **`5020`**
- `import: ../configs/tasks/task_assembly.yaml` → **`../../configs/...`**
  (relative to `generate_dataset/configs/`, not the repo root)

And in `run.py`: the assignment config is now loaded through the repo's
`ConfigLoader` (which resolves `import:` / `overwrite:` directives) instead of a
plain `yaml.safe_load`, which produced pydantic validation errors.

---

## 5. Completed steps — how to reproduce

Run everything from the repo root (`master_thesis/`) with the `master` env active.

### Step 0–3 — Patches & configs
Already in the files listed in §3/§4. Nothing to run.

### Step 4 — Verify the model API ✅

```bash
conda run -n master python -m generate_dataset.check_api
```

Verified output (2026-10-01):

```
Backend  : API
Endpoint : https://morpheus.cit.tum.de/api/v1
Model    : google/gemma-4-31B-it
API key  : SET ✅
[1/2] GET /models  → 200 OK
      Available : ['google/gemma-4-31B-it', 'cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4']
[2/2] POST /chat/completions → 200
      Response  : 'hello'
      usage     : {'prompt_tokens': 20, 'total_tokens': 22, 'completion_tokens': 2, ...}
API SANITY CHECK: PASSED ✅
```

> `usage` uses `completion_tokens`, not `output_tokens`. `config.USAGE_KEY_MAP`
> already maps this correctly.

### Step 5 — Build Docker images ✅

```bash
conda run -n master python -m generate_dataset.setup_docker --task os --skip-existing
```

Useful flags: `--dry-run`, `--task os|db|all`, `--skip-existing`.

Verified result — 7 images, ~5.4 GB on disk, ~4 min on a warm cache:

| Image | Size | Role |
|---|---|---|
| `ubuntu:latest` | 180 MB | base for `local-os/*` |
| `jingbh/agentrl-controller:latest` | 90 MB | FC controller |
| `redis:7` | 192 MB | sandbox locking / worker state |
| `local-os/default` | 972 MB | sandbox — **55** problems use it (+ default fallback) |
| `local-os/packages` | 1.97 GB | sandbox — 1 problem uses it |
| `local-os/ubuntu` | 176 MB | sandbox — unused by std data, built for parity |
| `agentbench/os-worker` | 1.81 GB | the `os-std` task worker |

Sandbox usage counts were measured from `data/os_interaction/data/**.json`
(`"local": "default"` ×55, `"local": "packages"` ×1).

**`mysql:8` + `agentbench/db-worker` are NOT needed** for os-std trace
generation — only for the `dbbench-std` task (`--task db`).

> The `local-os/*` Dockerfiles use `FROM docker.1ms.run/ubuntu` (a Chinese CDN
> mirror). `setup_docker.py` detects when that host is unreachable and rewrites
> the base to `ubuntu:latest` in a temp copy, leaving the originals untouched.

### Step 6 — Start the local server stack ✅

```bash
conda run -n master python -m generate_dataset.serve_local up
conda run -n master python -m generate_dataset.serve_local status
conda run -n master python -m generate_dataset.serve_local logs
conda run -n master python -m generate_dataset.serve_local down
```

Verified output:

```
✅  controller is up
✅  os-std worker registered
```

Confirmed working at this point:
- worker loads **144** `os-std` problem configs
- `GET /api/list_workers` reports the worker `ALIVE`, capacity 32
- `TaskClient('os-std').get_concurrency()` returns **32** (was 0 before the fix)
- a sample dispatches, a **sandbox container spawns**, init/start scripts run,
  and the loop reaches `Starting round 1/8` — i.e. it really does call the model

Worker log excerpt from a real dispatch:

```
[task.py:375]: Container initialized successfully
[task.py:510]: Execution setup completed successfully
[task.py:420]: Starting round 1/8
```

---

## 6. ✅ Blocker resolution (implemented option C)

### The problem

This repo's host-side client (`src/client/task.py`, `src/assigner.py`) speaks
the OLD AgentBench protocol and is incompatible with the shipped FC controller.

Measured against the live controller (2026-10-01):

```bash
curl -s -X POST http://localhost:5020/api/start_sample \
  -H 'Content-Type: application/json' \
  -d '{"name":"os-std","index":120}' -D -
```

```
HTTP/1.1 200 OK
Session_id: 8                     ← session id is in a HEADER
Content-Type: application/json

{"messages":[...], "tools":[...]}  ← body has only these two keys
```

What the legacy client expects instead:

```python
# src/client/task.py, run_sample()
sid = result["session_id"]                  # → KeyError: 'session_id'
while SampleStatus(result["output"]["status"]) == SampleStatus.RUNNING:
    agent.inference(result["output"]["history"])
```

So `smoke_test.py` fails with:

```
File "src/client/task.py", line 80, in run_sample
    sid = result["session_id"]
KeyError: 'session_id'
```

The whole loop (`/interact`, `/cancel`, `/calculate_overall`) is built on the old
body shape and will not line up either.

**Interpretation:** the FC/AgentRL version of AgentBench is meant to be driven by
**AgentRL's own harness**. The legacy `src/` client was left in the repo but never
updated for the new controller protocol.

### Solution implemented (option C)

We wrote a **purpose-built loop** (`agent_loop.py`) that talks directly to the
FC controller's HTTP API. No patching, no legacy client code.

**Protocol (measured 2026-10-14 against live controller):**

| Endpoint | Method | Session ID | Body | Response |
|---|---|---|---|---|
| `/api/start_sample` | POST | — | `{"name": task, "index": idx}` | **Header** `Session_id: <int>`<br>Body `{"messages": [...], "tools": [...]}` |
| `/api/interact` | POST | **Header** | `{"messages": [<assistant_msg>]}` | `{"finish": bool, "status": str, "messages": [<tool_results>], "metrics": {...}}` |
| `/api/cancel` | POST | **Header** | `{}` | `null` |

Key discoveries:
- Session ID goes in the **header**, not the body (both directions after `start_sample`)
- Controller returns empty `messages` for some samples (idx 19) — the worker failed to inject the initial prompt; fail fast instead of burning retries
- `finish: true` → controller drops the session immediately; another `interact` returns `{"message": "session not found"}`
- Gemma 4 31B supports tool calling properly: `finish_reason: "tool_calls"`, well-formed `tool_calls` array, `content: None` on tool turns

**Implementation:**

```python
# agent_loop.py
class ModelClient:
    """OpenAI-compatible chat with tool calling support."""
    def chat(messages, tools) -> (assistant_message, usage)

class ControllerClient:
    """FC controller HTTP API wrapper."""
    def start_sample(task, index) -> (session_id, messages, tools)
    def interact(session_id, assistant_message) -> step_response
    def cancel(session_id)

def run_sample(task, index, model, controller, round_limit=8) -> SampleResult:
    """
    start_sample → [ model.chat → controller.interact ] × N → finish/round_limit
    
    Returns: SampleResult with .passed, .history, .usage (all tokens/tools/turns captured)
    """
```

Token capture is exact because we own every model call — no patching needed.

---

## 7. Step 7 — Trace generation (complete ✅)

### 7a. Single sample smoke test

```bash
conda run -n master python -m generate_dataset.smoke_test
```

Verified output (2026-10-14):

```
✅ idx=0 status=completed rounds=3 tokens=1891 (73s)
   passed : True   (reward=1.0, score=1.0)
   usage  : in=1802 out=79 total=1881 tool_calls=3
SMOKE TEST: PASSED ✅
```

### 7b. Scale-up: 1 → 5 → 144 samples

```bash
# Phase 1: 1 sample
conda run -n master python -m generate_dataset.run --samples 1

# Phase 2: 5 samples at concurrency 2
conda run -n master python -m generate_dataset.run --samples 5 --concurrency 2

# Phase 3: all 144 at concurrency 4
conda run -n master python -m generate_dataset.run --all --concurrency 4
```

**Final result (2026-10-15):**

```
samples      : 144  (pass 73 / wrong 69 / errors 2)
accuracy     : 0.514
avg tokens   : in=3586 out=236 total=3822
avg tool calls: 4.32   avg turns: 4.32
wall clock   : 68 min (concurrency 4)
```

- **2 errors** (idx 19, 74) — controller returned empty conversations; skipped in post-processing
- **142 valid traces** saved
- Pass rate matches Gemma 4's actual capability on os-std (51.4% = 73/142)

### Output structure

```
master_thesis/
├── outputs/traces/
│   ├── traces.jsonl                                      ← 142 traces, mimo schema, ready for reward.py
│   └── 20261014_210315/gemma-local/os-std/
│       ├── runs.jsonl                                    ← 144 raw records (full history + usage)
│       └── overall.json                                  ← aggregate: acc, avg tokens, timing
└── reward_scores/
    └── gemma-4-31B-it/
        ├── all_results.json                              ← 142 per-trace rewards + components + metrics
        ├── summary.json                                  ← run metadata (judge model, timing, status)
        ├── input.json                                    ← staged JSON array (jsonl→array conversion)
        ├── items/                                        ← per-trace .json files (142 files)
        └── batches/                                      ← grouped batches of 10
```

**`traces.jsonl` line schema** (mimo-compatible, scored by `reward.py`):

```json
{
  "session_id": "0",
  "prompt": "<task description>",
  "messages": [
    {"role": "system|user|assistant|tool", "content": "...", "tool_calls": [...], "reasoning_content": "..."}
  ],
  "num_tool_calls": 4,
  "metadata": {
    "usage": {
      "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0
    }
  },
  "trace": [{"subtype": "summary", "durationMs": 95340}],
  "agentbench": {"result": true, "tool_calls": 4, "turns": 5, "task": "os-std", "index": 0}
}
```

Notes:
- `tool_calls` and `reasoning_content` are preserved on assistant messages — both are part of the mimo schema; `reward.py` uses them for its `redundancy_penalty` and `reasoning_efficiency` signals
- `trace[0].durationMs` carries wall-clock from `SampleResult.elapsed_s` so `reward.py`'s `duration_efficiency` signal has real data (not a free 1.0)

---

## 8. Step 8 — Scoring with reward.py (complete ✅)

### 8a. Wrapper script

`reward_scores/score_traces.py` converts `traces.jsonl` (JSON Lines) → `input.json`
(JSON array) and invokes `mimo-claude-code-traces-1k/reward.py`'s CLI. This
preserves `reward.py`'s resumable checkpointing and batch outputs.

### 8b. Smoke test (1 trace)

```bash
conda run -n master python -m reward_scores.score_traces --limit 1 --fresh --with-judge
```

Verified (2026-10-15): judge returns `llm_judge: 1.0`, overall reward 0.6357
(vs 0.5714 deterministic-only). Judge call took ~16s.

### 8c. Full scoring (142 traces with judge)

```bash
conda run -n master python -m reward_scores.score_traces --fresh --with-judge
```

**Completed:** 142/142 in ~37 min (judge enabled).

**Baseline reward distribution (Gemma 4 31B on AgentBench os-std):**

| Metric | Value |
|---|---|
| **Mean reward** | **0.5985** (median 0.6349, stdev 0.095) |
| Range | 0.34 → 0.73 |
| **Component means** | |
| token_efficiency | 0.27 (token-hungry: mean 3822 tokens/sample) |
| tool_efficiency | 0.71 (mean 4.3 calls, max 8 = round cap) |
| reasoning_efficiency | 1.00 (flat — Gemma 4 doesn't populate `reasoning_content`) |
| context_efficiency | 0.54 |
| redundancy_penalty | 0.97 (rarely repeats tool calls) |
| output_density | 0.53 |
| duration_efficiency | 0.11 (slow: mean 110s wall-clock) |
| **llm_judge** | **0.77** (Morpheus judge, Gemma 4 31B) |

**Top 3 traces** (efficient): 2-tool, ~1300-token solves, judge=1.0, reward ~0.72-0.73  
**Bottom 3** (inefficient): hit 8-round limit, 6-10k tokens, judge=0.1-0.3, reward ~0.34-0.41

All scored results in `reward_scores/gemma-4-31B-it/all_results.json`.

---

## 9. Quick reference

```bash
# 0. daemon (macOS, after reboot)
colima start

# 1. verify the model API
conda run -n master python -m generate_dataset.check_api

# 2. build images  (Step 5)
conda run -n master python -m generate_dataset.setup_docker --task os --skip-existing

# 3. start the stack  (Step 6)
conda run -n master python -m generate_dataset.serve_local up

# 4. one-sample smoke test  (Step 7a)
conda run -n master python -m generate_dataset.smoke_test

# 5. full trace generation  (Step 7b)
conda run -n master python -m generate_dataset.run --all --concurrency 4

# 6. score with LLM judge  (Step 8)
conda run -n master python -m reward_scores.score_traces --fresh --with-judge

# teardown
conda run -n master python -m generate_dataset.serve_local down
docker ps -aq --filter 'name=default-' | xargs -r docker rm -f   # orphaned sandboxes
```

### Switching model backend

Edit one line in `config.py`:

```python
MODEL_BACKEND: str = "api"     # "api" → Morpheus/OpenAI | "local" → vLLM
```

Or override per-run: `--backend api` / `--backend local`.

For `local`, start vLLM first:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-12B-it --tensor-parallel-size 2 --port 8000
```

---

## 10. Extending to other AgentBench tasks

The pipeline is task-agnostic. To add `dbbench-std` (300 samples, ~3-4h locally):

```bash
# 1. Build DB images
conda run -n master python -m generate_dataset.setup_docker --task db --skip-existing

# 2. Add mysql + db-worker to docker-compose.local.yml (see extra/docker-compose.yml)

# 3. Restart stack
conda run -n master python -m generate_dataset.serve_local down
conda run -n master python -m generate_dataset.serve_local up

# 4. Generate traces
conda run -n master python -m generate_dataset.run --task dbbench-std --all --concurrency 4

# 5. Score
conda run -n master python -m reward_scores.score_traces \
  --input outputs/traces/<TIMESTAMP>/gemma-local/dbbench-std/runs.jsonl \
  --output-dir reward_scores/dbbench-std --fresh --with-judge
```

**Other tasks:**

| Task | Samples | Setup notes | Recommended |
|---|---:|---|---|
| `dbbench-std` | 300 | `mysql:8` + db-worker (~800 MB) | ✅ local |
| `kg-std` | 150 | Freebase service (~50 GB data dump) | ⚠️ cluster |
| `webshop-std` | 200 | 16 GB RAM needed | ⚠️ cluster |
| `alfworld-std` | 6 | memory leak on long runs | ⚠️ cluster |

For cluster: use `docker compose -f extra/docker-compose.yml up --build` (full 5-task + freebase stack).

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker: command not found` | `brew install docker` + `colima start` |
| `the --mount option requires BuildKit` | `brew install docker-buildx` + `cliPluginsExtraDirs` in `~/.docker/config.json` |
| `docker: unknown command: compose` | `brew install docker-compose` + same `cliPluginsExtraDirs` |
| `ModuleNotFoundError: agentrl` on the host | Expected. `agentrl` only exists in the worker image; `patch.py` skips that patch. |
| `No such image: local-os/default` | Run Step 5 (`setup_docker.py`). |
| `redis.exceptions.ConnectionError ... 6379` | Redis service missing — it's in `docker-compose.local.yml`; bring the stack up with `serve_local.py`. |
| Assigner runs but dispatches nothing | The `get_concurrency` string/IntEnum bug (§4). Verify with `TaskClient('os-std','http://localhost:5020/api').get_concurrency()` — should be > 0. |
| `'int' object has no attribute 'connect'` | `aiodocker` pin (§4); rebuild `agentbench/os-worker`. |
| Traces generated but `reward.py` fails | Check trace schema — `tool_calls` / `reasoning_content` must be preserved on assistant messages; `trace[0].durationMs` needed for duration signal. |
| Judge scoring is slow | Expected — ~25s per trace × 142 ≈ 60 min. Deterministic-only is instant: omit `--with-judge`. |


---

## 12. Scaling to the full dataset (all 5 AgentBench tasks)

### 12.1 Task inventory

| Task | Samples | Round limit | Est. time (c=4) | Setup complexity | Location |
|---|---:|---:|---:|---|---|
| `os-std` ✅ | 144 | 8 | ~70 min | ✅ done | local/cluster |
| `dbbench-std` | 300 | 15 | ~3 h | mysql:8 + db-worker | local/cluster |
| `kg-std` | 150 | 15 | ~2 h | Freebase (~50 GB) | **cluster only** |
| `webshop-std` | 200 | 20 | ~3 h | 16 GB RAM | **cluster only** |
| `alfworld-std` | 6 | ~30 | ~5 min | memory leak | **cluster only** |
| **Total** | **800** | | **~9 h** | | |

Add ~2 h for judge scoring all 5 → **~11 h end-to-end** for the full dataset.

### 12.2 On the cluster: using a local vLLM server instead of Morpheus API

**Why:** Lower latency, no API rate limits, full control over model/quantization,
no per-token cost if the cluster already runs the model for other work.

**Prerequisites:**
- Cluster node with GPU(s) — Gemma 4 31B needs ~60 GB VRAM for full precision,
  or ~20-25 GB with AWQ/GPTQ quantization
- vLLM installed: `pip install vllm`

#### Step 1: Start the vLLM server on the cluster

```bash
# On a GPU node (e.g., via SLURM or directly if the node is dedicated)
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --port 8000 \
  --host 0.0.0.0
```

**Flags:**
- `--tensor-parallel-size 2`: split the model across 2 GPUs (adjust to your setup)
- `--port 8000`: expose on port 8000 (change if occupied)
- `--host 0.0.0.0`: listen on all interfaces so other nodes can reach it
- Add `--quantization awq` or `--quantization gptq` if using a quantized checkpoint

For a **quantized** model to save VRAM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model casperhansen/gemma-2-27b-it-awq \
  --quantization awq \
  --tensor-parallel-size 1 \
  --port 8000 \
  --host 0.0.0.0
```

> **Note:** Replace `casperhansen/gemma-2-27b-it-awq` with an actual Gemma 4 AWQ
> checkpoint if available, or use full-precision Gemma 2 27B as a stand-in.

Verify the server is up:

```bash
curl http://localhost:8000/v1/models
# Should return: {"object":"list","data":[{"id":"google/gemma-4-31B-it", ...}]}
```

#### Step 2: Configure `generate_dataset` to use the local server

Edit `master_thesis/generate_dataset/config.py`:

```python
MODEL_BACKEND: str = "local"  # ← change from "api" to "local"

LOCAL_BASE_URL = "http://localhost:8000/v1"  # ← vLLM endpoint
LOCAL_MODEL    = "google/gemma-4-31B-it"     # ← must match --model from vLLM
```

Or override at runtime without editing:

```bash
conda run -n master python -m generate_dataset.run \
  --backend local \
  --base-url http://localhost:8000/v1 \
  --model google/gemma-4-31B-it \
  --task os-std --all --concurrency 4
```

**If vLLM runs on a different cluster node** (e.g., GPU node `gpu01`, you're on
the login node), replace `localhost` with the node's hostname or IP:

```python
LOCAL_BASE_URL = "http://gpu01.cluster.example.edu:8000/v1"
```

Or via SSH tunnel from the login node:

```bash
# On your login node, forward port 8000 from gpu01 to local 8000
ssh -N -L 8000:localhost:8000 gpu01 &

# Then use localhost:8000 in config.py
LOCAL_BASE_URL = "http://localhost:8000/v1"
```

#### Step 3: Run trace generation

Same commands as before — the backend switch is transparent to `run.py`:

```bash
conda run -n master python -m generate_dataset.run --task os-std --all --concurrency 4
conda run -n master python -m generate_dataset.run --task dbbench-std --all --concurrency 4
# ... etc
```

**No API key needed** — vLLM's OpenAI-compatible server doesn't enforce auth by
default. If you added `--api-key` to the vLLM server, set it in `.env`:

```bash
echo "VLLM_API_KEY=your_key" >> .env
```

And update `config.py`:

```python
LOCAL_API_KEY_ENV = "VLLM_API_KEY"
```

#### Step 4: Batch all 5 tasks

On the cluster with the full stack (`docker compose -f extra/docker-compose.yml up`):

```bash
#!/bin/bash
# generate_all_traces.sh

TASKS=("os-std" "dbbench-std" "kg-std" "webshop-std" "alfworld-std")

for TASK in "${TASKS[@]}"; do
  echo "=== Generating traces for $TASK ==="
  conda run -n master python -m generate_dataset.run \
    --backend local \
    --task "$TASK" \
    --all \
    --concurrency 8 \
    || echo "❌ $TASK failed"
done

echo "=== All trace generation complete ==="
```

Run it:

```bash
chmod +x generate_all_traces.sh
./generate_all_traces.sh 2>&1 | tee full_traces.log
```

#### Step 5: Score all traces

```bash
#!/bin/bash
# score_all_traces.sh

TASKS=("os-std" "dbbench-std" "kg-std" "webshop-std" "alfworld-std")
TIMESTAMP="20261015_xxxxxx"  # ← replace with actual timestamp from outputs/traces/

for TASK in "${TASKS[@]}"; do
  echo "=== Scoring $TASK ==="
  conda run -n master python -m reward_scores.score_traces \
    --input "outputs/traces/$TIMESTAMP/gemma-local/$TASK/runs.jsonl" \
    --output-dir "reward_scores/$TASK" \
    --fresh \
    --with-judge \
    || echo "❌ $TASK scoring failed"
done

echo "=== All scoring complete ==="
```

Run it:

```bash
chmod +x score_all_traces.sh
./score_all_traces.sh 2>&1 | tee full_scoring.log
```

### 12.3 Cluster-specific setup (one-time)

#### Docker daemon

Most clusters don't run Colima — use the native Docker engine:

```bash
sudo systemctl start docker
sudo systemctl enable docker  # auto-start on reboot
sudo usermod -aG docker $USER  # allow non-root docker commands
# Log out and back in for the group change to take effect
```

Verify:

```bash
docker info
docker compose version
```

#### Build all images

```bash
conda run -n master python -m generate_dataset.setup_docker --task all --skip-existing
```

This builds:
- `local-os/*`, `agentbench/os-worker` (done)
- `agentbench/db-worker` + pulls `mysql:8`
- `agentbench/kg-worker` (needs the Freebase mount — see below)
- `agentbench/webshop-worker`
- `agentbench/alfworld-worker`

Total: ~8-10 GB across all images + ~50 GB for Freebase data.

#### Freebase setup (for `kg-std`)

The KG task needs a Virtuoso DB serving the Freebase dump. Clone the setup repo
**on the cluster**:

```bash
cd ~/
git clone https://github.com/THUDM/Freebase-Setup
cd Freebase-Setup
# Follow its README to download the Freebase dump (~50 GB compressed, ~180 GB uncompressed)
# and start the Virtuoso container
```

Then update `extra/docker-compose.yml` to mount the Freebase service URL into
the `kg-worker` environment (already configured in the shipped compose file).

#### Start the full stack

```bash
cd master_thesis
docker compose -f extra/docker-compose.yml up --build -d
```

Wait ~2 min for all workers to register:

```bash
conda run -n master python -m generate_dataset.serve_local status
# Should show all 5 workers registered
```

### 12.4 Comparison: API vs local vLLM

| | Morpheus API | Local vLLM |
|---|---|---|
| **Setup** | `.env` with API key | vLLM server + GPU |
| **Latency** | ~20-25s per sample | ~10-15s (on-node) |
| **Cost** | per-token metered | GPU time (amortized) |
| **Concurrency** | limited by rate limits | limited by GPU memory |
| **Model control** | fixed (Gemma 4 31B) | any model, quantization |
| **Best for** | prototyping, small runs | production, full dataset |

For 800 samples × 4k tokens ≈ **3.2M tokens** → Morpheus cost is negligible for
a one-time baseline. But if you're iterating (re-running after fine-tuning, trying
different prompts), local vLLM wins.

### 12.5 Quick reference: full dataset generation

```bash
# === CLUSTER SETUP (one-time) ===
# 1. Start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --port 8000 \
  --host 0.0.0.0

# 2. Build all task images
conda run -n master python -m generate_dataset.setup_docker --task all

# 3. Start full AgentBench stack
docker compose -f extra/docker-compose.yml up -d

# === GENERATE TRACES (all 5 tasks, ~9h) ===
for TASK in os-std dbbench-std kg-std webshop-std alfworld-std; do
  conda run -n master python -m generate_dataset.run \
    --backend local \
    --task $TASK --all --concurrency 8
done

# === SCORE (all 5 tasks with judge, ~2h) ===
TS=$(ls -td outputs/traces/*/gemma-local | head -1 | cut -d/ -f3)
for TASK in os-std dbbench-std kg-std webshop-std alfworld-std; do
  conda run -n master python -m reward_scores.score_traces \
    --input outputs/traces/$TS/gemma-local/$TASK/runs.jsonl \
    --output-dir reward_scores/$TASK \
    --fresh --with-judge
done

# === RESULTS ===
# 800 traces in outputs/traces/traces.jsonl
# Per-task scored results in reward_scores/{task}/all_results.json
```
