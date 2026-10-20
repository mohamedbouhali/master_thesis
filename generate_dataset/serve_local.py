"""
generate_dataset/serve_local.py
===============================
Step 6 (LOCAL) — bring up the minimal AgentBench server stack for testing:
one controller + one os_interaction worker, via Colima/Docker.

This is the local-laptop counterpart of the cluster's
`docker compose -f extra/docker-compose.yml up`.  It uses the trimmed
`generate_dataset/docker-compose.local.yml` (OS task only).

Commands:
    up      build (if needed) and start controller + os worker, wait until the
            worker registers with the controller
    down    stop and remove the stack
    status  show controller + worker registration state
    logs    tail the stack logs

Usage (from master_thesis/ root):
    conda run -n master python -m generate_dataset.serve_local up
    conda run -n master python -m generate_dataset.serve_local status
    conda run -n master python -m generate_dataset.serve_local down

Notes
-----
- The controller serves BOTH the workers and the assigner client on :5020/api.
  (The old README/docs mention :5000 — that was AgentBench v0.2 and is stale
  for this function-calling / AgentRL version.)
- Requires Docker + BuildKit (buildx).  On macOS, start Colima first:
      colima start
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT   = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "generate_dataset" / "docker-compose.local.yml"
CONTROLLER_URL = "http://localhost:5020/api"


def _compose(*args: str, env: dict | None = None) -> int:
    """Run `docker compose -f <local> <args>` from the repo root."""
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    print(f"  $ {' '.join(cmd)}")
    full_env = dict(os.environ)
    full_env["DOCKER_BUILDKIT"] = "1"
    full_env["COMPOSE_DOCKER_CLI_BUILD"] = "1"
    if env:
        full_env.update(env)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=full_env).returncode


def _controller_up() -> bool:
    try:
        r = requests.get(f"{CONTROLLER_URL}/list_workers", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _os_worker_registered() -> bool:
    """True once an os-std worker has registered with the controller."""
    try:
        r = requests.get(f"{CONTROLLER_URL}/list_workers", timeout=5)
        if r.status_code != 200:
            return False
        workers = r.json()
        info = workers.get("os-std")
        if not info:
            return False
        # at least one ALIVE worker with capacity
        for w in info.get("workers", {}).values():
            if w.get("status") in ("ALIVE", "alive"):
                return True
        return False
    except Exception:
        return False


def cmd_up() -> int:
    print("=" * 55)
    print("  Starting local AgentBench stack (controller + os worker)")
    print("=" * 55)

    rc = _compose("up", "--build", "-d")
    if rc != 0:
        print("\n❌  docker compose up failed. Is Colima running?  (colima start)")
        return rc

    print("\n  Waiting for controller on :5020 ...")
    for _ in range(30):
        if _controller_up():
            print("  ✅  controller is up")
            break
        time.sleep(2)
    else:
        print("  ❌  controller did not come up in time. Check: serve_local logs")
        return 1

    print("  Waiting for os-std worker to register (can take ~1 min) ...")
    for _ in range(60):
        if _os_worker_registered():
            print("  ✅  os-std worker registered")
            print("\n  Stack ready. Next:")
            print("    conda run -n master python -m generate_dataset.run --task os-std")
            return 0
        time.sleep(2)

    print("  ⚠️   worker not registered yet. Check logs:")
    print("    conda run -n master python -m generate_dataset.serve_local logs")
    return 1


def cmd_down() -> int:
    print("  Stopping local AgentBench stack ...")
    return _compose("down", "-v")


def cmd_status() -> int:
    print("Controller :5020 :", "UP ✅" if _controller_up() else "DOWN ❌")
    print("os-std worker    :", "REGISTERED ✅" if _os_worker_registered() else "NOT REGISTERED ❌")
    print()
    return _compose("ps")


def cmd_logs() -> int:
    return _compose("logs", "--tail", "80")


def main():
    parser = argparse.ArgumentParser(
        description="Manage the local AgentBench test stack (controller + os worker)."
    )
    parser.add_argument(
        "command",
        choices=["up", "down", "status", "logs"],
        help="up | down | status | logs",
    )
    args = parser.parse_args()

    if args.command == "up":
        sys.exit(cmd_up())
    elif args.command == "down":
        sys.exit(cmd_down())
    elif args.command == "status":
        sys.exit(cmd_status())
    elif args.command == "logs":
        sys.exit(cmd_logs())


if __name__ == "__main__":
    main()
