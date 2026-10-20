# generate_dataset — Progress & Reproduction Guide

**Goal:** run the AgentBench FC `os-std` task with a language model served over an
**OpenAI-compatible API** (TUM Morpheus / Gemma 4), capture **token-efficiency
metrics** per sample, and **save the resulting traces locally** in a schema that
`mimo-claude-code-traces-1k/reward.py` can score directly.

**Status:** Steps 0–6 done and verified. Step 7 (full run) is **blocked** by a
protocol mismatch between this repo's legacy host client and the shipped FC
controller — see [Blocker](#-blocker-must-be-resolved-before-step-7).

Last updated: 2026-10-01

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
| `smoke_test.py` | Runs **one** `os-std` sample end-to-end. This is what surfaced the blocker. | ⚠️ reveals blocker |
| `run.py` | Step 7 — full run + post-processing to `traces.jsonl`. | ❌ blocked |
| `patch.py` | Monkey-patches for token capture (no repo source edits). Host: patches `HTTPAgent`. Worker: also patches `OSInteraction`. Skips the latter gracefully when `agentrl` is absent. | ✅ imports |
| `configs/local-gemma.yaml` | `HTTPAgent` agent definition (url/headers/body/prompter). | ✅ |
| `configs/assignment.yaml` | Assignment skeleton; `run.py` injects url/model/key at runtime. | ✅ |

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

## 6. ⛔ BLOCKER — must be resolved before Step 7

**This repo's host-side client (`src/client/task.py`, `src/assigner.py`) speaks
the OLD AgentBench protocol and is incompatible with the shipped FC controller.**

Measured against the live controller:

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
updated for the new controller protocol. This is not a configuration mistake on
our side.

### Options (pick one before writing more code)

| | Approach | Pros | Cons |
|---|---|---|---|
| **A** | Retrofit `src/client/task.py` to the FC protocol (read `Session_id` header, use `messages`/`tools`, adapt `interact`) | keeps `src.assigner` (concurrency, resume, retries) | editing legacy code we don't own; large surface |
| **B** | Drive the controller with AgentRL's own eval entry point | most "intended" path | unknown API; adds a heavy dependency; less control over trace output |
| **C** | **Write a small purpose-built loop** against the controller's HTTP API (`start_sample` → model → `interact` → …), reusing our `HTTPAgent` + usage capture | full control over the trace schema and token capture; small and testable; no legacy baggage | we reimplement scheduling/retries (but we only need simple sequential/bounded concurrency) |

**Recommendation: C.** The goal is *trace generation with token metrics*, not
reproducing AgentBench leaderboard numbers. A ~150-line loop gives exactly the
output schema `reward.py` needs, and the token-usage capture becomes trivial
because we own both sides of the call.

### Note on where metrics get captured

`patch.py` patches two places:
- **host**: `HTTPAgent.inference()` → stores `last_usage` from the API response
- **worker**: `OSInteraction` → accumulates per-turn usage into the result

With option **C** the host-side capture is sufficient, since our loop makes every
model call itself — the worker-side patch becomes optional.

---

## 7. Next steps

1. **Decide A / B / C** above. (Recommended: **C**.)
2. Implement the chosen loop in `generate_dataset/` (new module, e.g. `agent_loop.py`),
   reusing `HTTPAgent` + `patch.py`'s usage capture.
3. Wire it into `run.py` so the existing `post_process_traces()` still produces
   `traces.jsonl`.
4. Run a 1-sample smoke test, then a 5-sample run, then the full 144.
5. Score the output:
   ```bash
   python mimo-claude-code-traces-1k/reward.py \
     --input master_thesis/outputs/traces/traces.jsonl
   ```
6. Once local works, move to the cluster: same `setup_docker.py`, but
   `docker compose -f extra/docker-compose.yml up --build` for the full stack.

### Target output (unchanged by the option chosen)

```
outputs/traces/<TIMESTAMP>/gemma-local/os-std/
    runs.jsonl      # raw per-sample output
    overall.json    # pass/fail  +  efficiency aggregate
outputs/traces/
    traces.jsonl    # flat, mimo-schema, ready for reward.py
```

`traces.jsonl` line schema (what `post_process_traces()` emits):

```json
{
  "session_id": "0",
  "prompt": "<task description>",
  "messages": [{"role": "user|assistant|tool", "content": "..."}],
  "num_tool_calls": 4,
  "metadata": {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}},
  "agentbench": {"result": true, "tool_calls": 4, "turns": 5,
                 "task": "os-std", "index": 0}
}
```

---

## 8. Quick reference

```bash
# 0. daemon (macOS, after reboot)
colima start

# 1. verify the model API
conda run -n master python -m generate_dataset.check_api

# 2. build images  (Step 5)
conda run -n master python -m generate_dataset.setup_docker --task os --skip-existing

# 3. start the stack  (Step 6)
conda run -n master python -m generate_dataset.serve_local up

# 4. one-sample end-to-end  (currently hits the blocker)
conda run -n master python -m generate_dataset.smoke_test

# 5. full run  (blocked until §6 is resolved)
conda run -n master python -m generate_dataset.run --task os-std --concurrency 1 --auto-retry

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

### Troubleshooting

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
| `KeyError: 'session_id'` | **The blocker** (§6). |
