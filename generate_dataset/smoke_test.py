"""
generate_dataset/smoke_test.py
==============================
End-to-end smoke test for the FC agent loop: run N os-std samples against the
local AgentBench stack + the configured model backend, and print results.

Verifies the whole chain:
    ModelClient (Morpheus/vLLM) ↔ ControllerClient ↔ os worker ↔ sandbox

Prereqs:
    - local stack up:   python -m generate_dataset.serve_local up
    - images built:     python -m generate_dataset.setup_docker --task os
    - API key in .env (loaded automatically)

Usage (from master_thesis/ root):
    conda run -n master python -m generate_dataset.smoke_test
    conda run -n master python -m generate_dataset.smoke_test --samples 3
    conda run -n master python -m generate_dataset.smoke_test --index 5 --verbose
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "generate_dataset"))

# Load API key from .env (repo root first, then the mimo dataset repo)
for _env in (REPO_ROOT / ".env",
             REPO_ROOT.parent / "mimo-claude-code-traces-1k" / ".env"):
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import config as cfg                                      # noqa: E402
from agent_loop import ControllerClient, ModelClient, run_sample   # noqa: E402

CONTROLLER_URL = "http://localhost:5020/api"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the FC agent loop.")
    parser.add_argument("--samples", type=int, default=1,
                        help="How many samples to run (default: 1).")
    parser.add_argument("--index", default=None,
                        help="Run one specific sample index instead.")
    parser.add_argument("--task", default=cfg.TASK_NAME)
    parser.add_argument("--round-limit", type=int, default=8)
    parser.add_argument("--verbose", action="store_true",
                        help="Print the full conversation.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    base_url = cfg.resolved_base_url()
    model_id = cfg.resolved_model()
    api_key = cfg.resolved_api_key()

    print("=" * 58)
    print(f"  backend    : {cfg.MODEL_BACKEND}")
    print(f"  model      : {model_id}")
    print(f"  endpoint   : {base_url}")
    print(f"  controller : {CONTROLLER_URL}")
    print("=" * 58)

    controller = ControllerClient(CONTROLLER_URL)
    model = ModelClient(
        base_url=base_url,
        model=model_id,
        api_key=api_key,
        temperature=cfg.TEMPERATURE,
        max_tokens=cfg.MAX_TOKENS,
        timeout=cfg.TIMEOUT_S,
    )

    try:
        indices = controller.list_indices(args.task)
    except Exception as exc:
        print(f"\n❌  Cannot reach controller: {exc}")
        print("    Start the stack: python -m generate_dataset.serve_local up")
        return 1

    capacity = controller.worker_capacity(args.task)
    print(f"  task       : {args.task}  ({len(indices)} samples, capacity {capacity})")
    if capacity == 0:
        print("\n❌  No worker capacity — is the os worker registered?")
        print("    Check: python -m generate_dataset.serve_local status")
        return 1

    if args.index is not None:
        # argparse gives a string, but the controller's indices are typed
        # (ints for os-std) and it rejects a mismatched type with
        # "task ... does not support index N". Resolve against the real list.
        match = [i for i in indices if str(i) == str(args.index)]
        if not match:
            print(f"\n❌  index {args.index!r} is not in the task's index list.")
            return 1
        targets = match
    else:
        targets = indices[: args.samples]
    print(f"  running    : {targets}\n")

    results = []
    for idx in targets:
        print(f"── sample {idx} " + "─" * 40)
        res = run_sample(args.task, idx, model, controller, round_limit=args.round_limit)
        results.append(res)

        print(f"   status : {res.status}")
        print(f"   passed : {res.passed}   (reward={res.reward}, score={res.score})")
        print(f"   rounds : {res.rounds}")
        print(f"   usage  : in={res.usage['input_tokens']} "
              f"out={res.usage['output_tokens']} "
              f"total={res.usage['total_tokens']} "
              f"tool_calls={res.usage['tool_calls']}")
        print(f"   history: {len(res.history)} messages")
        print(f"   time   : {res.elapsed_s}s")
        if res.error:
            print(f"   error  : {res.error}")

        if args.verbose:
            print("   ── conversation ──")
            for m in res.history:
                role = m.get("role")
                content = str(m.get("content") or "")[:160].replace("\n", " ")
                tcs = m.get("tool_calls") or []
                names = [t.get("function", {}).get("name") for t in tcs]
                extra = f"  tool_calls={names}" if names else ""
                print(f"     [{role}] {content}{extra}")
        print()

    ok = all(r.error is None for r in results)
    n_pass = sum(1 for r in results if r.passed)
    tok = sum(r.usage["total_tokens"] for r in results)

    print("=" * 58)
    print(f"  {len(results)} sample(s): {n_pass} passed, "
          f"{len(results) - n_pass} failed/unsolved")
    print(f"  total tokens: {tok}")
    print("=" * 58)
    print("SMOKE TEST: PASSED ✅" if ok else "SMOKE TEST: FAILED ❌ (see errors above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
