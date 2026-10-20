"""
reward_scores/score_traces.py
=============================
Score the generated AgentBench traces with mimo-claude-code-traces-1k/reward.py
and save the results next to this file.

Why a wrapper
-------------
`reward.py` expects a JSON **array** and writes to a specific layout
(items/, batches/, summary.json, all_results.json). Our generator writes
JSON **Lines** (one trace per line). This script:
  1. Converts traces.jsonl → traces.json (JSON array)   [idempotent, no-op if up-to-date]
  2. Invokes reward.py's own CLI pointed at reward_scores/

Running reward.py's CLI (not calling compute_reward() ourselves) means we
inherit its:
  - resumable checkpointing (safe to kill and resume)
  - per-item + batch outputs for inspection
  - clean summary.json / all_results.json

Usage (from master_thesis/ root)
--------------------------------
    conda run -n master python -m reward_scores.score_traces --limit 1
    conda run -n master python -m reward_scores.score_traces              # all 144
    conda run -n master python -m reward_scores.score_traces --with-judge # also hit Morpheus
    conda run -n master python -m reward_scores.score_traces --fresh      # ignore existing checkpoints
"""

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR    = Path(__file__).resolve().parent
REPO_ROOT   = THIS_DIR.parent                               # master_thesis/
TRACES_JSONL = REPO_ROOT / "outputs" / "traces" / "traces.jsonl"
REWARD_PY    = REPO_ROOT.parent / "mimo-claude-code-traces-1k" / "reward.py"


def _load_env(path: Path) -> None:
    """Load KEY=VALUE lines without overriding existing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def jsonl_to_json_array(jsonl: Path, out_json: Path, limit: int | None = None) -> int:
    """Convert JSON-Lines → JSON array. Drops lines with errors / empty history."""
    records = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Skip traces that didn't produce a conversation (e.g. the two errored
        # samples): reward.py would score them as zeros across every signal.
        if not rec.get("messages"):
            continue
        records.append(rec)
        if limit and len(records) >= limit:
            break

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(records)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", type=Path, default=TRACES_JSONL,
                   help=f"traces.jsonl to score (default: {TRACES_JSONL}).")
    p.add_argument("--output-dir", type=Path, default=THIS_DIR / "gemma-4-31B-it",
                   help="Where to write per-item + summary outputs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N traces (useful for a smoke test).")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--with-judge", action="store_true",
                   help="Enable the LLM-as-judge signal (extra API cost).")
    p.add_argument("--model", default="google/gemma-4-31B-it",
                   help="Judge model (used only with --with-judge).")
    p.add_argument("--base-url", default="https://morpheus.cit.tum.de/api/v1",
                   help="Judge endpoint (used only with --with-judge).")
    p.add_argument("--api-key-env", default="MORPHEUS_API_KEY")
    p.add_argument("--fresh", action="store_true",
                   help="Delete existing checkpoints in --output-dir before scoring.")
    args = p.parse_args()

    if not args.input.exists():
        print(f"❌  input not found: {args.input}")
        return 1
    if not REWARD_PY.exists():
        print(f"❌  reward.py not found at {REWARD_PY}")
        return 1

    # ── load env (so MORPHEUS_API_KEY is available for --with-judge) ──
    _load_env(REPO_ROOT / ".env")
    _load_env(REWARD_PY.parent / ".env")

    # ── convert jsonl → json array ──
    staged = args.output_dir / "input.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh and args.output_dir.exists():
        for child in ("items", "batches"):
            d = args.output_dir / child
            if d.exists():
                shutil.rmtree(d)
        for name in ("progress.json", "summary.json", "all_results.json"):
            f = args.output_dir / name
            if f.exists():
                f.unlink()

    n = jsonl_to_json_array(args.input, staged, limit=args.limit)
    print(f"staged {n} traces → {staged}")

    # ── invoke reward.py's CLI ──
    cmd = [
        sys.executable, str(REWARD_PY),
        "--input",       str(staged),
        "--output-dir",  str(args.output_dir),
        "--batch-size",  str(args.batch_size),
        "--api-key-env", args.api_key_env,
        "--model",       args.model,
        "--base-url",    args.base_url,
    ]
    if args.with_judge:
        cmd.append("--with-judge")

    print("$", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n❌  reward.py exited with code {rc}")
        return rc

    # ── human-friendly top-line summary ──
    summary = args.output_dir / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        print("\n" + "=" * 55)
        print(f"  scored     : {s.get('completed_items')}/{s.get('total_items')}")
        print(f"  failed     : {s.get('failed_this_run', 0)}")
        print(f"  avg item   : {s.get('average_new_item_seconds', 0)}s")
        print(f"  judge      : {'ON' if s.get('judge', {}).get('enabled') else 'off'}")
        print(f"  outputs    : {args.output_dir}")
        print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
