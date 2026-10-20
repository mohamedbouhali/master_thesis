# AgentBench FC — Summary

## 1. What is AgentBench FC?

AgentBench (Function Calling version) is a **live, containerized benchmark** for
evaluating LLMs as agents. It runs your model against real tasks inside Docker
environments and produces objective pass/fail results per task sample.

It is integrated with **AgentRL**, an end-to-end multi-task, multi-turn LLM
agent reinforcement learning framework.

---

## 2. Supported Tasks

| Task ID | Domain | Notes |
|---|---|---|
| `os-std` | OS / Linux shell | Closest to coding domain |
| `dbbench-std` | SQL / Database | |
| `knowledgegraph` | Knowledge graph queries | Requires Freebase data |
| `alfworld` | Embodied household tasks | ~16GB RAM |
| `webshop` | Web shopping | ~16GB RAM |

All tasks run inside Docker containers. The lite preset (`os-std` + `dbbench-std`,
1 worker each) is the recommended starting point on a laptop or for initial runs.

---

## 3. What You Provide vs What the Framework Provides

| You provide | AgentBench provides |
|---|---|
| Local model served via vLLM | System prompt (hardcoded per task) |
| 4-line YAML config (url, model name) | Tool definitions |
| Nothing else | Task descriptions (the problems) |
| | Tool execution (bash / SQL in Docker) |
| | Pass/fail evaluation |
| | Full multi-turn interaction loop |
| | Session management |
| | Conversation history |
| | Concurrency across workers |
| | Output persistence |

---

## 4. Your Model's Interface

Your model receives a plain list of messages on every turn and returns the next
assistant message. That is the only interface.

```
[system, user, assistant, tool_result, assistant, tool_result, ...]
                    ↓
          return next assistant message
                    ↓
          framework handles the rest
```

The framework:
- maintains the full conversation history across turns
- appends tool results after each execution
- passes the updated history to your model on the next turn
- terminates the session when the model calls a finish/answer tool or hits the round limit

---

## 5. Agent Configuration

The agent is configured by a single YAML file. For a local vLLM model:

```yaml
# configs/agents/local-chat.yaml
module: src.client.agents.HTTPAgent
parameters:
  url: http://localhost:8000/v1/chat/completions
  headers:
    Content-Type: application/json
    Authorization: Bearer not-needed
  body:
    model: google/gemma-4-12B-it
    temperature: 0
    max_tokens: 1024
  prompter:
    name: role_content_dict
    args:
      agent_role: assistant
  return_format: "{response[choices][0][message][content]}"
```

No OpenAI key needed. vLLM exposes an OpenAI-compatible endpoint that AgentBench's
`HTTPAgent` calls directly.

---

## 6. Tools

### 6.1 os_interaction — 3 tools

```yaml
bash_action(script: string)       # execute a bash command in the Docker container
finish_action(thought: string)    # signal the task is done (no answer required)
answer_action(answer: string)     # submit a direct answer to a question
```

**Round limit:** 8 turns per task sample.

### 6.2 dbbench — 2 tools

```yaml
execute_sql(query: string)        # run a SQL query on the database
commit_final_answer(answers: [])  # submit the final answer(s)
```

**Round limit:** 15 turns per task sample.

### 6.3 Complexity

Each tool takes a single string (or array) parameter. No nested objects, no
optional fields, no complex schemas. The model only needs to reliably emit one
JSON tool call per turn.

### 6.4 Adding New Tools

Tools are declared in `configs/tasks/<task>.yaml` and handled in
`src/server/tasks/<task>/task.py`. Adding a tool is two steps:

1. Add the function definition to the YAML `tools:` list.
2. Handle the new `func_name` in `_extract_function()` and `_handle_agent_action()`
   in `task.py`.

---

## 7. How to Run

### Prerequisites
- Docker running
- Python 3.9 (pins old numpy)
- Port 5000–5015 free (on macOS: disable AirPlay Receiver)

### Install
```bash
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
```

### Build Docker images (os-std + dbbench-std)
```bash
docker pull mysql
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default \
             data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages \
             data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu \
             data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

### Start task server (terminal 1)
```bash
python -m src.start_task -a --config configs/start_task_lite.yaml
# wait ~1 min for "200 OK"
```

### Start assigner (terminal 2)
```bash
python -m src.assigner --config configs/assignments/lite.yaml
```

Results are written to `outputs/{TIMESTAMP}/`.

---

## 8. Output Format

Per agent+task pair, two files are written:

```
outputs/{TIMESTAMP}/{agent}/{task}/
  ├── results.jsonl       # per-sample result (index, status, history)
  └── overall.json        # aggregate stats
```

Default `overall.json` (pass/fail only):
```json
{
  "overall": {
    "total": 50,
    "pass": 38,
    "wrong": 12,
    "acc": 0.76
  }
}
```

---

## 9. Adding a Custom Efficiency Reward (3-file patch)

AgentBench only tracks pass/fail by default. To also capture token usage and
tool-call efficiency for thesis evaluation, 3 small changes are needed.

> **Why this is easy with a local model:** vLLM always returns a `usage` field
> in every response (`prompt_tokens`, `completion_tokens`, `total_tokens`).
> AgentBench currently ignores it — the patch just captures it.

### 9.1 `src/client/agents/http_agent.py`
Capture `usage` from the vLLM response after each inference call:
```python
resp = resp.json()
self.last_usage = resp.get("usage", {})  # add this line
return self.return_format.format(response=resp)
```

### 9.2 `src/server/tasks/os_interaction/task.py`
Accumulate usage per turn during `_handle_agent_action`, then attach totals to
`result.result` at the end of `_judge`:
```python
result.result["usage"] = {
    "total_input_tokens": ...,
    "total_output_tokens": ...,
    "total_tool_calls": ...,
    "turns": ...
}
```

### 9.3 `src/server/tasks/os_interaction/task.py` — `calculate_overall`
Compute efficiency stats across all completed samples:
```python
overall["avg_input_tokens"]  = mean(r.result["usage"]["total_input_tokens"] ...)
overall["avg_output_tokens"] = mean(r.result["usage"]["total_output_tokens"] ...)
overall["avg_tool_calls"]    = mean(r.result["usage"]["total_tool_calls"] ...)
overall["avg_turns"]         = mean(r.result["usage"]["turns"] ...)
```

### Resulting `overall.json` after patch
```json
{
  "overall": { "total": 50, "pass": 38, "acc": 0.76 },
  "efficiency": {
    "avg_input_tokens": 3100,
    "avg_output_tokens": 420,
    "avg_tool_calls": 4.8,
    "avg_turns": 3.9
  }
}
```

This gives both axes needed for the thesis:
**task success rate** (correctness) + **token/tool usage** (efficiency).

---

## 10. Role in the Thesis Pipeline

```
mimo-claude traces  ──►  reward.py (efficiency scores)  ──►  SFT fine-tuning
  (offline training)                                       (Gemma 4 12B + LoRA)
                                                                   │
                                                                   ▼
                                               vLLM local endpoint (no API key)
                                                                   │
                                                                   ▼
                                          AgentBench os-std (live evaluation)
                                                                   │
                                                         ┌─────────┴─────────┐
                                                   task success          token usage
                                                  (pass/fail acc)    (patched overall.json)
```

- **Training source:** mimo-claude coding traces (offline, efficiency-shaped SFT)
- **Live evaluation harness:** AgentBench `os-std` (correctness + token efficiency
  after the 3-file patch)
- **Thesis claim:** fine-tuned Gemma 4 reduces token usage while maintaining
  `acc` within an acceptable delta vs baseline

> Note: domain mismatch exists (mimo = coding tasks, AgentBench = OS/DB/KG).
> `os_interaction` is the closest AgentBench task to the training domain.
> A coding-native live benchmark (e.g. SWE-bench Verified) would give stronger
> domain alignment — pending advisor confirmation.
