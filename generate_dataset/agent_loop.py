"""
generate_dataset/agent_loop.py
==============================
Purpose-built agent loop for the AgentBench **FC / AgentRL** controller
(option "C" from progress.md).

Why this exists
---------------
This repo's legacy host client (`src/client/task.py`, `src/assigner.py`) speaks
the OLD AgentBench protocol and is incompatible with the shipped FC controller
(`jingbh/agentrl-controller`). It crashes on `result["session_id"]` because the
new controller returns the session id in a **header**, not the body.

Rather than retrofit the legacy assigner, this module talks to the controller
directly. We own both sides of every model call, which makes token-usage
capture exact instead of patched-in.

Measured controller protocol (probed 2026-10-14 against the live controller)
---------------------------------------------------------------------------
POST /api/start_sample
    body     : {"name": "<task>", "index": <idx>}
    response : header `Session_id: <int>`
               body   {"messages": [...], "tools": [...]}

POST /api/interact
    header   : Session_id: <int>
    body     : {"messages": [<assistant message>]}
    response : {"finish": bool,
                "reward": <num>,
                "status": "running" | "completed" | ...,
                "messages": [<tool result messages>],
                "metrics": {"score": <num>}}

POST /api/cancel
    header   : Session_id: <int>
    body     : {}
    response : null

Notes:
  * After `finish: true` the session is dropped — interacting again returns
    `{"message": "session not found"}`.
  * `metrics.score` / `reward` carry the pass/fail signal (1 = solved).
  * Sending the session id in the body instead of the header yields
    `{"message": "invalid session id"}`.

Model requirements
------------------
The FC protocol needs real **tool calls** from the model, so we cannot reuse
`HTTPAgent` (its `return_format` collapses the response to a content string and
discards `tool_calls`). `ModelClient` below returns the full assistant message
plus the `usage` block.

Verified: `google/gemma-4-31B-it` on Morpheus returns well-formed `tool_calls`
with `finish_reason: "tool_calls"`. On tool turns `content` is `None`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger("generate_dataset.agent_loop")

# Fields the model may return on an assistant message that we must NOT echo
# back to the API (some providers reject unknown/null extras).
_ASSISTANT_PASSTHROUGH = ("role", "content", "tool_calls", "reasoning_content")


# ---------------------------------------------------------------------------
# Model client
# ---------------------------------------------------------------------------

class ModelClient:
    """
    Minimal OpenAI-compatible chat client that preserves tool calls and usage.

    Works against any OpenAI-shaped endpoint — Morpheus, OpenAI, or a local
    vLLM server — since all three expose POST {base_url}/chat/completions.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> tuple[dict, dict]:
        """
        Send one chat completion request.

        Returns
        -------
        (assistant_message, usage)
            assistant_message : dict with role/content/tool_calls (API shape)
            usage             : {"prompt_tokens", "completion_tokens", "total_tokens"}

        Raises on persistent failure.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _sanitize_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout
                )
            except Exception as exc:                      # network-level failure
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    message = data["choices"][0]["message"]
                    usage = data.get("usage") or {}
                    return message, usage
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"

            if attempt < self.max_retries - 1:
                sleep_s = 2 ** attempt
                logger.warning(
                    "model call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.max_retries, last_error, sleep_s,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"model call failed after {self.max_retries} attempts: {last_error}")


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """
    Strip provider-specific extras (refusal/annotations/audio/function_call/
    reasoning) before echoing a conversation back to the API, keeping only the
    fields the chat API understands.
    """
    cleaned: list[dict] = []
    for msg in messages:
        out = {"role": msg.get("role")}

        content = msg.get("content")
        # An assistant turn that only called tools has content=None; that is
        # valid, but keep it explicit so the API sees the right shape.
        out["content"] = content if content is not None else (
            None if msg.get("tool_calls") else ""
        )

        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            out["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            out["name"] = msg["name"]

        cleaned.append(out)
    return cleaned


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Outcome of one task sample, including the full trace and token usage."""

    index: Any
    status: str = "unknown"          # completed | task_limit_reached | error | ...
    passed: bool = False             # derived from reward / metrics.score
    reward: float = 0.0
    score: float = 0.0
    rounds: int = 0
    history: list[dict] = field(default_factory=list)   # full conversation
    usage: dict = field(default_factory=dict)           # token + structural metrics
    error: Optional[str] = None
    elapsed_s: float = 0.0

    def to_runs_record(self) -> dict:
        """
        Serialize into the shape `run.py::post_process_traces()` consumes, so
        the downstream traces.jsonl contract is unchanged:

            {"index": ..., "output": {"status", "result", "history"}, "time": {...}}
        """
        ts_ms = int(time.time() * 1000)
        return {
            "index": self.index,
            "output": {
                "status": self.status,
                "result": {
                    "result": self.passed,
                    "reward": self.reward,
                    "score": self.score,
                    "rounds": self.rounds,
                    # wall-clock, surfaced so reward.py's duration_efficiency
                    # signal has real data instead of defaulting to 0 ms
                    "elapsed_s": self.elapsed_s,
                    "usage": self.usage,
                },
                "history": self.history,
            },
            "error": self.error,
            "time": {
                "timestamp": ts_ms,
                "str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000)),
            },
        }


# ---------------------------------------------------------------------------
# Controller client
# ---------------------------------------------------------------------------

class ControllerClient:
    """Thin HTTP client for the FC controller's session API."""

    def __init__(self, controller_url: str, timeout: float = 300.0) -> None:
        self.base = controller_url.rstrip("/")
        self.timeout = timeout

    def list_indices(self, task: str) -> list:
        r = requests.get(f"{self.base}/get_indices", params={"name": task}, timeout=30)
        r.raise_for_status()
        return r.json()

    def worker_capacity(self, task: str) -> int:
        """Free capacity across ALIVE workers for a task (0 if none)."""
        try:
            r = requests.get(f"{self.base}/list_workers", timeout=30)
            r.raise_for_status()
            info = r.json().get(task)
        except Exception:
            return 0
        if not info:
            return 0
        free = 0
        for w in info.get("workers", {}).values():
            # controller reports status as the string "ALIVE"
            if str(w.get("status")).upper() == "ALIVE" and not w.get("locked", False):
                free += max(0, w.get("capacity", 0) - w.get("current", 0))
        return free

    def start_sample(self, task: str, index: Any) -> tuple[str, list[dict], list[dict]]:
        """
        Returns (session_id, messages, tools).
        The session id arrives in the `Session_id` response header.
        """
        r = requests.post(
            f"{self.base}/start_sample",
            json={"name": task, "index": index},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"start_sample failed [{r.status_code}]: {r.text[:300]}")

        session_id = r.headers.get("Session_id") or r.headers.get("session_id")
        if not session_id:
            raise RuntimeError(
                f"start_sample returned no Session_id header; headers={dict(r.headers)}"
            )
        body = r.json()
        return session_id, body.get("messages", []), body.get("tools", [])

    def interact(self, session_id: str, assistant_message: dict) -> dict:
        """Submit one assistant turn; returns the controller's step response."""
        r = requests.post(
            f"{self.base}/interact",
            headers={"Content-Type": "application/json", "Session_id": str(session_id)},
            json={"messages": [assistant_message]},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"interact failed [{r.status_code}]: {r.text[:300]}")
        data = r.json()
        if isinstance(data, dict) and "message" in data and "status" not in data:
            # e.g. {"message": "session not found"} / {"message": "invalid session id"}
            raise RuntimeError(f"interact rejected: {data['message']}")
        return data

    def cancel(self, session_id: str) -> None:
        """Best-effort session cleanup; never raises."""
        try:
            requests.post(
                f"{self.base}/cancel",
                headers={"Content-Type": "application/json", "Session_id": str(session_id)},
                json={},
                timeout=30,
            )
        except Exception as exc:
            logger.debug("cancel(%s) failed: %s", session_id, exc)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_sample(
    task: str,
    index: Any,
    model: ModelClient,
    controller: ControllerClient,
    round_limit: int = 8,
) -> SampleResult:
    """
    Run ONE task sample end-to-end:

        start_sample  →  [ model → interact ] × N  →  finish / round limit

    Accumulates token usage per turn and returns the full conversation.
    `round_limit` should match the task config (os.yaml uses 8); the controller
    also enforces its own limit, whichever trips first.
    """
    started = time.perf_counter()
    result = SampleResult(index=index)

    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_calls": 0,
        "turns": 0,
        "model_calls": 0,
    }

    session_id: Optional[str] = None
    try:
        session_id, messages, tools = controller.start_sample(task, index)
        conversation = list(messages)       # system + initial user message

        for round_no in range(round_limit):
            assistant_msg, usage = model.chat(conversation, tools)

            # ── accumulate token usage (OpenAI naming → our naming) ──
            usage_totals["input_tokens"] += usage.get("prompt_tokens", 0) or 0
            usage_totals["output_tokens"] += usage.get("completion_tokens", 0) or 0
            usage_totals["total_tokens"] += usage.get("total_tokens", 0) or 0
            usage_totals["model_calls"] += 1
            usage_totals["turns"] += 1
            usage_totals["tool_calls"] += len(assistant_msg.get("tool_calls") or [])

            # keep only the fields that belong in a trace
            trace_msg = {k: assistant_msg[k] for k in _ASSISTANT_PASSTHROUGH
                         if k in assistant_msg}
            trace_msg.setdefault("role", "assistant")
            # some providers put chain-of-thought in `reasoning`
            if "reasoning" in assistant_msg and assistant_msg["reasoning"]:
                trace_msg["reasoning_content"] = assistant_msg["reasoning"]
            conversation.append(trace_msg)

            step = controller.interact(session_id, trace_msg)

            tool_messages = step.get("messages") or []
            conversation.extend(tool_messages)

            result.rounds = round_no + 1
            result.status = step.get("status", result.status)
            result.reward = float(step.get("reward", 0) or 0)
            result.score = float((step.get("metrics") or {}).get("score", 0) or 0)

            if step.get("finish"):
                session_id = None          # controller already dropped it
                break
        else:
            # loop exhausted without `finish`
            result.status = result.status or "round_limit"
            logger.info("sample %s hit the local round limit (%d)", index, round_limit)

        result.history = conversation
        result.passed = (result.score > 0) or (result.reward > 0)

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.status = "error"
        logger.warning("sample %s failed: %s", index, result.error)
    finally:
        if session_id:                      # unfinished → free the sandbox
            controller.cancel(session_id)

    result.usage = usage_totals
    result.elapsed_s = round(time.perf_counter() - started, 2)
    return result
