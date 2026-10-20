# Master Thesis Project Context

This steering file captures long-term context for the master thesis. It is a
living document; update it as decisions are made.

## Repositories in the workspace

- `master_thesis/` — This is **AgentBench** (THUDM), the function-calling (FC)
  version integrated with AgentRL. It evaluates LLMs-as-agents across
  containerized tasks: `alfworld` (AF), `dbbench` (DB), `knowledgegraph` (KG),
  `os_interaction` (OS), `webshop` (WS). Tasks run in Docker; the agent talks to
  an OpenAI-compatible HTTP endpoint.
- `mimo-claude-code-traces-1k/` — Dataset + reward pipeline for scoring agent
  traces on efficiency (token/tool/reasoning efficiency, LLM-as-judge). Has
  `reward.py`, `run_reward_all.py`, `run_reward_one.py`, `load_dataset.py`.
- `CodeTracer/` — NJU-LINK/Kwaipilot trajectory diagnosis system (`codetracer`
  CLI). Separate tool, not the primary thesis repo.

## How AgentBench runs (key mechanics)

- Agent is defined by `configs/agents/*.yaml`. The default is
  `gpt-3.5-turbo-0613` (deprecated OpenAI model) wired in
  `configs/agents/api_agents.yaml` -> `openai-chat.yaml`.
- The agent client is `src.client.agents.HTTPAgent` (`src/client/agents/http_agent.py`).
  It just POSTs an OpenAI-style chat/completions request to a configurable `url`
  with configurable `headers` and `body`. This means **any OpenAI-compatible
  endpoint works** — including a locally hosted model server.
- Task server: `python -m src.start_task -a [--config configs/start_task_lite.yaml]`
  Assigner (runs the eval): `python -m src.assigner [--config configs/assignments/lite.yaml]`
- Lite presets exist for laptops / limited RAM (1 worker, only dbbench-std + os-std).
- Requires Docker running. README recommends Python 3.9 (pins old numpy).

## Environment notes (user is on macOS)

- macOS AirPlay Receiver occupies port 5000; task server assumes ports 5000-5015
  free. Disable AirPlay Receiver or free the port.
- `webshop` needs ~16GB RAM; `alfworld` leaks memory. Prefer `dbbench-std` and
  `os-std` for light runs.

## Thesis goal (north star)

**Optimize the agent to use fewer tokens while keeping the same task
performance.** Efficiency-focused agent optimization, measured on AgentBench
task success and on the efficiency reward signals in `mimo-claude-code-traces-1k/reward.py`.

## Pipeline / plan

1. Run a small language model **locally** in place of the hosted LLM. Serve it
   via an OpenAI-compatible endpoint (vLLM / Ollama / llama.cpp / LM Studio) and
   point an AgentBench agent config at that local `url`. No OpenAI key needed
   (dummy Authorization header or none).
2. **Generate traces** by running the agent over the dataset / AgentBench tasks.
3. **Score traces** with `reward.py` (token/tool/reasoning/context efficiency,
   redundancy, output density, duration, optional LLM-as-judge). Reward is a
   scalar in [0,1]; weights configured in `WEIGHTS`.
4. **Optimize** the agent to reduce token usage while holding task success
   constant (the two axes: keep AgentBench success rate, push efficiency reward up).

## Key design tension: OFFLINE data vs LIVE environment

The two candidate resources differ fundamentally in whether they let you *run*
a fine-tuned agent and measure real task success:

- **mimo-claude-code-traces-1k** = OFFLINE dataset of 1,017 static Claude-Code
  traces. NO environment, NO task verifier, NO ground-truth pass/fail. Tasks ran
  in an agentic coding sandbox that is NOT shipped. `reward.py` here scores
  *efficiency only* (token/tool/reasoning/etc.); the optional LLM-judge also
  only rates *efficiency*, not correctness. So this dataset alone CANNOT verify
  that a fine-tuned agent still solves tasks — it can only score efficiency of
  traces, and can only be used for offline training (SFT/DPO/reward-weighted).
- **AgentBench (master_thesis)** = LIVE environment. Dockerized tasks with real
  success/failure verification (dbbench, os_interaction, kg, alfworld, webshop).
  You CAN drop in a fine-tuned model, run it, and get an objective task-success
  score. This is what makes "keep same performance" measurable.

Implication: the mimo dataset is a *training/distillation* source; AgentBench is
the *evaluation harness*. They are complementary, not either/or. The natural
thesis architecture: train on mimo (offline, efficiency-shaped) -> evaluate the
fine-tuned agent live on AgentBench (success rate + token cost).

Caveat on transfer: mimo traces are Claude-Code coding tasks; AgentBench FC
tasks are DB/OS/KG/etc. Domain/tool mismatch means skills learned on mimo may
not transfer cleanly to AgentBench tasks. Options to address: (a) evaluate only
on AgentBench tasks closest to coding (os_interaction), (b) hold out a mimo test
split for efficiency eval and use AgentBench only as a correctness sanity check,
(c) consider a coding-native live benchmark (e.g. SWE-bench) if closer alignment
is needed.

## Confirmed decisions (2026-09-16)

1. **Thesis contribution = the METHOD**: making agents token-efficient. The
   training/optimization technique is the centerpiece, not the evaluation. The
   eval exists to prove the method preserves task performance.
2. **Domain alignment matters** (user wants strong, credible results). Training
   domain and evaluation domain should match: coding -> coding. This weakens the
   case for AgentBench FC (DB/OS/KG/ALFWorld/WebShop) as the primary eval, since
   only os_interaction is coding-adjacent. Prefer a coding-native LIVE benchmark
   with real pass/fail verification (e.g. SWE-bench / SWE-bench Verified, or a
   similar executable coding benchmark) as the correctness harness.

Consequence: mimo-claude-code-traces-1k stays the offline TRAINING source
(coding traces, efficiency reward). The live EVALUATION should be coding-native.
AgentBench may still serve as a secondary/cross-domain robustness check, but is
no longer the main evaluation target. Revisit whether master_thesis (AgentBench)
remains the working repo or whether a coding benchmark harness gets added.

## Compute + model (confirmed 2026-09-16)

- **Cluster**: 2x AMD EPYC 7662 (128 cores total), 1TB DDR4 RAM,
  8x NVIDIA A100 40GB (=320GB aggregate VRAM). Strong single-node multi-GPU box.
- **Model**: Gemma 4 (Google, released ~Mar/Apr 2026, Apache-2.0). Family:
  dense E2B (2.3B), E4B (4.5B), 12B, 31B; plus 26B-A4B MoE (3.8B activated,
  26B total). Native function-calling / agentic support (good for tool-use agent).
  HF ids like `google/gemma-4-12B-it`, `google/gemma-4-26B-A4B`, `google/gemma-4-31B-it`.
  NOTE: reward.py judge default `google/gemma-4-31B-it` is therefore a REAL id.

### Fit on 8xA100-40GB
- Inference (vLLM, bf16, tensor-parallel): 12B fits easily on 1-2 GPUs; 31B fits
  in bf16 across ~2-4 GPUs (~62GB weights + KV cache). 26B-A4B MoE also fine.
- Fine-tuning: full FT of 31B is tight on 40GB cards (needs ZeRO-3/FSDP offload,
  doable given 1TB RAM). LoRA/QLoRA on 12B or 31B is comfortable. Recommend
  starting with 12B + LoRA for fast iteration, scale to 31B if results warrant.
- Serving for AgentBench/eval: run vLLM OpenAI-compatible server, point HTTPAgent
  config `url` at it. No OpenAI key needed.

## Open questions to resolve

- Which Gemma 4 size to center on (recommend 12B for iteration, 31B for final).
- Serving runtime = vLLM (best throughput on A100 + OpenAI-compatible API).
- Which coding-native live benchmark to adopt for correctness eval (SWE-bench
  Verified is the strongest candidate; confirm compute/setup feasibility).
- Optimization method: **most likely SFT** (leaning reward-weighted SFT on the
  efficient mimo traces). DPO/RL possible later as improvement chapters. Not
  fully locked in yet.
- Definition of "same performance": exact benchmark + acceptable success delta
  (e.g. within X% of baseline resolve rate) alongside target token reduction.

## Cluster access

- Access is via **plain SSH** to the cluster (scheduler/SLURM not confirmed yet
  — ask). Plan launch scripts to work over SSH; adapt to SLURM if present.
- Gemma 4 is a **gated** HF model — will need `huggingface-cli login` + accepted
  license on the cluster before download. Confirm whether weights are already
  present there.

## Work already completed (found 2026-09-16)

- **Offline efficiency scoring of the mimo TRAIN split is DONE.** All 915 train
  traces scored via `reward.py` with the LLM judge ENABLED
  (`google/gemma-4-31B-it` @ TUM Morpheus endpoint `https://morpheus.cit.tum.de/api/v1`).
  0 failures. Ran ~6.3h (Sept 2-3, 2026, ~24.8s/item). Outputs in
  `mimo-claude-code-traces-1k/results/rewards/train/`:
  `all_results.json` (per-item reward+components+metrics), `summary.json`,
  `progress.json`, `items/`, `batches/`.
- Reward distribution looks healthy (real spread, e.g. 0.40 / 0.44 / 0.66 on
  first items; per-signal breakdowns + populated llm_judge). Ready to filter/rank
  for SFT.
- TEST split (`datasets/.../test.json`) is NOT yet scored (would be the held-out
  efficiency eval set).
- Note: TUM Morpheus endpoint served the judge — so the user has cluster/endpoint
  API access there. `MORPHEUS_API_KEY` used via .env.

## Status of pending items (as of 2026-09-16)

- [leaning SFT] Optimization method — user leans SFT, not final.
- [ASK ADVISOR] Live eval benchmark — user will ask advisor before committing
  (SWE-bench Verified is our working candidate). DO NOT build the eval harness
  until this is confirmed.
- Cluster is SSH-accessible; scheduler unknown.
- NEXT natural step (no blockers): build the SFT data-prep step — filter/rank the
  already-scored 915 train traces by reward, convert to chat SFT format for
  Gemma 4. This depends on nothing that's still open.

## CRITICAL FINDING (2026-10-01): legacy client vs FC controller protocol mismatch

While wiring the local end-to-end run of generate_dataset, discovered that this
repo's **host-side `src/client/task.py` + `src/assigner.py` are from the OLD
AgentBench protocol and are INCOMPATIBLE with the shipped FC controller image**
(`jingbh/agentrl-controller`).

Evidence (local Colima test against the real controller on :5020):
- Old client expects `start_sample` → body `{"session_id":..., "output":{"status","history"}}`.
- New FC controller actually returns body `{"messages":[...], "tools":[...]}`
  and puts the session id in the **`Session_id` HTTP response header**.
- `TaskClient.run_sample` crashes at `result["session_id"]` (KeyError).
- The whole interaction loop (interact/cancel/calculate_overall) uses the old
  body shape and will not line up with the FC controller either.

Interpretation: the FC/AgentRL version of AgentBench is meant to be driven by
**AgentRL's own harness** (the RL framework), not the legacy `src.assigner`.
The legacy client in `src/` was left in the repo but not updated for the new
controller protocol.

### What DOES work (verified locally)
- API backend to Morpheus (`google/gemma-4-31B-it`) — chat + usage field. ✅
- Docker stack: controller + redis + os worker, worker registers. ✅
- Sandbox container spawns, init/start scripts run, round loop reaches the model. ✅
- Fixes applied to get this far:
  * `src/client/agents/__init__.py`: made FastChatAgent import lazy (torch optional).
  * `src/server/tasks/os_interaction/task.py`: fixed a regex SyntaxWarning.
  * `src/client/task.py` `get_concurrency`: accept string "ALIVE" status (new
    controller returns a string, old IntEnum compare always gave 0).
  * `src/server/tasks/os_interaction/Dockerfile`: pin `aiodocker==0.24.0`
    (0.27 + current aiohttp raised "'int' object has no attribute 'connect'"
    when exec-ing in the sandbox).
  * generate_dataset: added serve_local.py, docker-compose.local.yml (controller
    + redis + os worker), check_api.py, smoke_test.py; fixed assignment.yaml
    controller port 5000→5020 and task_assembly import path; run.py now loads
    the assignment via ConfigLoader (import/overwrite resolution) + .env loader.

### The remaining blocker (needs a decision)
To run end-to-end, ONE of:
  (A) Rewrite the host client to speak the FC controller protocol (read
      `Session_id` header; use the messages/tools body; adapt interact + the
      finish/usage handling). Most faithful to "use AgentBench as the eval".
  (B) Drive the controller with AgentRL's own eval entry point instead of
      `src.assigner` (investigate what AgentRL exposes).
  (C) Build the agent loop ourselves against the controller's documented HTTP
      API (start_sample → model → interact → …), reusing our HTTPAgent + the
      token-usage capture. This is close to (A) but a purpose-built small loop
      rather than retrofitting the legacy assigner.

Token/efficiency capture still needs the OSInteraction-side patch to run INSIDE
the worker container (agentrl present there), not just on the host — our patch.py
already handles both contexts, but the metric must be surfaced through whatever
protocol path we choose above.

NEXT SESSION: decide A/B/C before writing more code. Local Colima stack is a
working reproduction to develop against.
