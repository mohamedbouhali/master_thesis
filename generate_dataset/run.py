"""
generate_dataset/run.py
=======================
Step 7 — generate AgentBench os-std traces with token/efficiency metrics and
save them locally.

Pipeline
--------
    for each task sample:
        agent_loop.run_sample()        # start_sample → model ↔ interact → finish
            ↓
    runs.jsonl        (raw per-sample: status, pass/fail, usage, full history)
            ↓
    overall.json      (aggregate: acc + efficiency averages)
            ↓
    traces.jsonl      (flat mimo schema — ready for reward.py)

Why not src.assigner
--------------------
This repo's legacy client speaks the OLD AgentBench protocol and is incompatible
with the shipped FC controller (it crashes on `result["session_id"]`; the FC
controller returns the session id in a header). We drive the controller directly
via `agent_loop.py` — see progress.md §6 and the protocol notes in agent_loop.py.

Backend
-------
Set `MODEL_BACKEND` in config.py ("api" | "local"), or override with --backend.
The API key is read from MORPHEUS_API_KEY (env, or a .env file — see below).

Usage (from master_thesis/ root)
--------------------------------
    # smoke: a single sample
    conda run -n master python -m generate_dataset.run --samples 1

    # a small batch
    conda run -n master python -m generate_dataset.run --samples 5 --concurrency 2

    # everything (144 os-std samples)
    conda run -n master python -m generate_dataset.run --all --concurrency 4

    # resume an interrupted run (skips samples already in runs.jsonl)
    conda run -n master python -m generate_dataset.run --all --resume-from outputs/traces/<TS>

Prereqs
-------
    python -m generate_dataset.setup_docker --task os   # images  (Step 5)
    python -m generate_dataset.serve_local up           # stack   (Step 6)
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Path setup — make sure we can import from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "generate_dataset"))


# ---------------------------------------------------------------------------
# Load API key from a .env file. Checks, in order:
#   1. master_thesis/.env
#   2. mimo-claude-code-traces-1k/.env   (where MORPHEUS_API_KEY currently lives)
# Env vars already set take precedence (setdefault).
# ---------------------------------------------------------------------------
def _load_env_files() -> None:
    for env_path in (
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "mimo-claude-code-traces-1k" / ".env",
    ):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_files()

import config as cfg                                            # noqa: E402
from agent_loop import ControllerClient, ModelClient, run_sample  # noqa: E402

CONTROLLER_URL = "http://localhost:5020/api"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_dataset")

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Endpoint check
# ---------------------------------------------------------------------------

def check_endpoint_reachable(base_url: str, model: str, api_key: str, backend: str) -> None:
    """Verify the model endpoint is up and the requested model is available."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(f"{base_url}/models", headers=headers, timeout=15)
        resp.raise_for_status()
        available = [m["id"] for m in resp.json().get("data", [])]
    except Exception as exc:
        hint = (
            "Is vLLM running?\n"
            f"  python -m vllm.entrypoints.openai.api_server --model {model} --port 8000"
            if backend == "local"
            else f"Check the endpoint and API key for {base_url}"
        )
        raise RuntimeError(
            f"Cannot reach model endpoint at {base_url}.\n{hint}\n\nOriginal error: {exc}"
        ) from exc

    if model not in available:
        raise RuntimeError(
            f"Model '{model}' not found at {base_url}.\n"
            f"Available: {available}\n"
            f"Update LOCAL_MODEL / API_MODEL in config.py."
        )
    logger.info("[%s] endpoint reachable — model '%s' available.", backend.upper(), model)


# ---------------------------------------------------------------------------
# Post-processing: runs.jsonl → mimo-compatible traces.jsonl
# ---------------------------------------------------------------------------

def post_process_traces(run_dir: Path, output_file: Path) -> int:
    """
    Convert runs.jsonl into a flat traces.jsonl matching the
    mimo-claude-code-traces-1k schema, so reward.py scores it unmodified.

    Output schema per line:
    {
      "session_id":     str,
      "prompt":         str,
      "messages":       [{"role", "content", ...}],
      "num_tool_calls": int,
      "metadata": {"usage": {input_tokens, output_tokens, total_tokens,
                             cache_read_input_tokens, cache_creation_input_tokens}},
      "agentbench":     {result, tool_calls, turns, task, index}   # extra, ignored by reward.py
    }

    `tool_calls` and `reasoning_content` are carried through on assistant
    messages when present — both are part of the mimo message schema and
    reward.py uses them for its redundancy_penalty and reasoning_efficiency
    signals. Dropping them would silently zero those components.
    """
    runs_file = run_dir / "runs.jsonl"
    if not runs_file.exists():
        logger.warning("runs.jsonl not found at %s — skipping post-processing.", runs_file)
        return 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(runs_file) as f_in, open(output_file, "a", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            output      = raw.get("output", {}) or {}
            history     = output.get("history", []) or []
            result_dict = output.get("result", {}) or {}
            usage       = result_dict.get("usage", {}) or {}
            index       = raw.get("index")

            # Normalize history → mimo message format.
            # (AgentBench v0.2 used role="agent"; the FC path already emits
            # "assistant", but we normalize defensively.)
            messages = []
            for item in history:
                role = item.get("role", "")
                if role == "agent":
                    role = "assistant"
                msg = {"role": role, "content": item.get("content", "")}
                if item.get("tool_calls"):
                    msg["tool_calls"] = item["tool_calls"]
                if item.get("reasoning_content"):
                    msg["reasoning_content"] = item["reasoning_content"]
                if item.get("tool_call_id"):
                    msg["tool_call_id"] = item["tool_call_id"]
                messages.append(msg)

            # Extract the task description from the first user message
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    text = str(msg.get("content") or "")
                    if "My problem is:" in text:
                        prompt = text.split("My problem is:", 1)[-1].strip()
                    else:
                        prompt = text
                    break

            trace = {
                "session_id":     str(index),
                "prompt":         prompt,
                "messages":       messages,
                "num_tool_calls": usage.get("tool_calls", 0),
                "metadata": {
                    "usage": {
                        "input_tokens":                usage.get("input_tokens",  0),
                        "output_tokens":               usage.get("output_tokens", 0),
                        "total_tokens":                usage.get("total_tokens",  0),
                        # neither vLLM nor the Morpheus endpoint reports cache stats
                        "cache_read_input_tokens":     0,
                        "cache_creation_input_tokens": 0,
                    }
                },
                # reward.py reads wall-clock from a trace event with
                # subtype == "summary" (durationMs). Emitting it keeps the
                # duration_efficiency signal meaningful rather than a free 1.0.
                "trace": [
                    {
                        "subtype":    "summary",
                        "durationMs": int(float(result_dict.get("elapsed_s", 0) or 0) * 1000),
                    }
                ],
                "agentbench": {
                    "result":     bool(result_dict.get("result", False)),
                    "tool_calls": usage.get("tool_calls", 0),
                    "turns":      usage.get("turns", 0),
                    "task":       run_dir.name,
                    "index":      index,
                },
            }

            f_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Post-processed %d samples → %s", written, output_file)
    return written


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_overall(results: list) -> dict:
    """Aggregate pass/fail accuracy plus the efficiency metrics."""
    done = [r for r in results if r.error is None]
    total = len(results)
    passed = sum(1 for r in done if r.passed)

    def _avg(key: str) -> float:
        vals = [r.usage.get(key, 0) for r in done]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "overall": {
            "total":   total,
            "pass":    passed,
            "wrong":   len(done) - passed,
            "errors":  total - len(done),
            "acc":     round(passed / len(done), 4) if done else 0.0,
        },
        "efficiency": {
            "avg_input_tokens":  _avg("input_tokens"),
            "avg_output_tokens": _avg("output_tokens"),
            "avg_total_tokens":  _avg("total_tokens"),
            "avg_tool_calls":    _avg("tool_calls"),
            "avg_turns":         _avg("turns"),
            "sum_total_tokens":  sum(r.usage.get("total_tokens", 0) for r in done),
            "avg_seconds":       round(
                sum(r.elapsed_s for r in done) / len(done), 2
            ) if done else 0.0,
        },
        "samples": [
            {
                "index":        r.index,
                "status":       r.status,
                "passed":       r.passed,
                "rounds":       r.rounds,
                "input_tokens":  r.usage.get("input_tokens", 0),
                "output_tokens": r.usage.get("output_tokens", 0),
                "total_tokens":  r.usage.get("total_tokens", 0),
                "tool_calls":    r.usage.get("tool_calls", 0),
                "seconds":       r.elapsed_s,
                "error":         r.error,
            }
            for r in results
        ],
    }


def already_done(run_dir: Path) -> set:
    """Indices already present in runs.jsonl (for --resume)."""
    runs_file = run_dir / "runs.jsonl"
    done = set()
    if not runs_file.exists():
        return done
    for line in runs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("error") is None and rec.get("index") is not None:
            done.add(rec["index"])
    return done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate AgentBench os-std traces with token-efficiency metrics."
    )
    p.add_argument("--backend", choices=["local", "api"], default=None,
                   help="Override MODEL_BACKEND from config.py.")
    p.add_argument("--model", default=None, help="Override the model name.")
    p.add_argument("--base-url", default=None, help="Override the API base URL.")
    p.add_argument("--api-key", default=None, help="Override the API key.")
    p.add_argument("--task", default=cfg.TASK_NAME, help=f"Task id (default: {cfg.TASK_NAME}).")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--samples", type=int, default=1,
                   help="Run the first N samples (default: 1).")
    g.add_argument("--all", action="store_true", help="Run every sample in the task.")
    g.add_argument("--index", default=None, help="Run one specific sample index.")

    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel samples (default: 1). Keep modest: each uses a sandbox "
                        "container and an API slot.")
    p.add_argument("--round-limit", type=int, default=8,
                   help="Max agent rounds per sample (os.yaml uses 8).")
    p.add_argument("--agent-name", default=cfg.AGENT_NAME)
    p.add_argument("--output-dir", type=Path, default=cfg.OUTPUT_DIR)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Reuse an existing run directory and skip completed samples.")
    p.add_argument("--controller", default=CONTROLLER_URL)
    p.add_argument("--skip-endpoint-check", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    backend  = args.backend  or cfg.MODEL_BACKEND
    base_url = args.base_url or cfg.resolved_base_url()
    model_id = args.model    or cfg.resolved_model()
    api_key  = args.api_key  or cfg.resolved_api_key()

    if backend == "api" and not api_key:
        logger.error(
            "API backend needs a key. Set %s in the environment or a .env file, "
            "or pass --api-key.", cfg.API_KEY_ENV,
        )
        return 1

    # Run directory (timestamped, or reused for --resume-from)
    if args.resume_from:
        run_root = args.resume_from
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_root = args.output_dir / ts
    run_dir = run_root / args.agent_name / args.task
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("generate_dataset — trace generation")
    logger.info("  backend     : %s", backend.upper())
    logger.info("  model       : %s", model_id)
    logger.info("  endpoint    : %s", base_url)
    logger.info("  controller  : %s", args.controller)
    logger.info("  task        : %s", args.task)
    logger.info("  concurrency : %d", args.concurrency)
    logger.info("  output      : %s", run_dir)
    logger.info("=" * 60)

    if not args.skip_endpoint_check:
        try:
            check_endpoint_reachable(base_url, model_id, api_key, backend)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1

    if args.controller == "enroot":
        from enroot_backend import EnrootController
        controller = EnrootController(round_limit=args.round_limit)
    else:
        controller = ControllerClient(args.controller)
    model = ModelClient(
        base_url=base_url,
        model=model_id,
        api_key=api_key,
        temperature=cfg.TEMPERATURE,
        max_tokens=cfg.MAX_TOKENS,
        timeout=cfg.TIMEOUT_S,
    )

    # Discover samples
    try:
        indices = controller.list_indices(args.task)
    except Exception as exc:
        logger.error("Cannot reach the controller at %s: %s", args.controller, exc)
        logger.error("Start the stack: python -m generate_dataset.serve_local up")
        return 1

    capacity = controller.worker_capacity(args.task)
    logger.info("task has %d samples; worker capacity %d", len(indices), capacity)
    if capacity == 0:
        logger.error("No worker capacity for '%s'. Check: "
                     "python -m generate_dataset.serve_local status", args.task)
        return 1

    if args.index is not None:
        # argparse yields a string; the controller's indices are typed (ints
        # for os-std) and it rejects a mismatch with "does not support index".
        match = [i for i in indices if str(i) == str(args.index)]
        if not match:
            logger.error("index %r is not in the task's index list.", args.index)
            return 1
        targets = match
    elif args.all:
        targets = list(indices)
    else:
        targets = list(indices)[: args.samples]

    if args.resume_from:
        done = already_done(run_dir)
        before = len(targets)
        targets = [t for t in targets if t not in done]
        logger.info("resume: %d already done, %d remaining", before - len(targets), len(targets))

    if not targets:
        logger.info("Nothing to run.")
        return 0

    effective_conc = max(1, min(args.concurrency, capacity, len(targets)))
    if effective_conc != args.concurrency:
        logger.info("concurrency clamped to %d", effective_conc)

    # ── run ──
    runs_file = run_dir / "runs.jsonl"
    results = []
    t0 = time.perf_counter()

    def _one(idx):
        return run_sample(args.task, idx, model, controller, round_limit=args.round_limit)

    with ThreadPoolExecutor(max_workers=effective_conc) as pool:
        futures = {pool.submit(_one, idx): idx for idx in targets}
        for n, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:                 # pragma: no cover
                logger.error("sample %s crashed: %s", idx, exc)
                continue

            results.append(res)
            # append incrementally so an interrupted run stays resumable
            with _write_lock, open(runs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res.to_runs_record(), ensure_ascii=False) + "\n")

            flag = "✅" if res.passed else ("⚠️ " if res.error is None else "❌")
            logger.info(
                "[%d/%d] %s idx=%s status=%s rounds=%d tokens=%d (%.0fs)%s",
                n, len(targets), flag, res.index, res.status, res.rounds,
                res.usage.get("total_tokens", 0), res.elapsed_s,
                f" error={res.error}" if res.error else "",
            )

    elapsed = time.perf_counter() - t0

    # ── aggregate ──
    overall = build_overall(results)
    overall["run"] = {
        "backend": backend,
        "model": model_id,
        "base_url": base_url,
        "task": args.task,
        "concurrency": effective_conc,
        "round_limit": args.round_limit,
        "elapsed_seconds": round(elapsed, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ── traces.jsonl (mimo schema) ──
    traces_file = args.output_dir / "traces.jsonl"
    n_traces = post_process_traces(run_dir, traces_file)

    o = overall["overall"]
    e = overall["efficiency"]
    logger.info("=" * 60)
    logger.info("  samples      : %d  (pass %d / wrong %d / errors %d)",
                o["total"], o["pass"], o["wrong"], o["errors"])
    logger.info("  accuracy     : %.3f", o["acc"])
    logger.info("  avg tokens   : in=%.0f out=%.0f total=%.0f",
                e["avg_input_tokens"], e["avg_output_tokens"], e["avg_total_tokens"])
    logger.info("  avg tool calls: %.2f   avg turns: %.2f",
                e["avg_tool_calls"], e["avg_turns"])
    logger.info("  wall clock   : %.0fs", elapsed)
    logger.info("=" * 60)
    logger.info("overall.json : %s", run_dir / "overall.json")
    logger.info("runs.jsonl   : %s", runs_file)
    logger.info("traces.jsonl : %s  (%d traces)", traces_file, n_traces)
    logger.info("Score them:\n  python mimo-claude-code-traces-1k/reward.py --input %s",
                traces_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
