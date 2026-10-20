"""
generate_dataset/setup_docker.py
=================================
Step 5 — Build and pull every Docker image required to run AgentBench
(os-std and dbbench-std tasks).

Fully reproducible: run this script on any machine with Docker installed
and it will produce exactly the images expected by the task workers.

Run from master_thesis/ root:
    conda run -n master python -m generate_dataset.setup_docker

    # Dry-run (print what would be done, no Docker calls):
    conda run -n master python -m generate_dataset.setup_docker --dry-run

    # Only os_interaction images:
    conda run -n master python -m generate_dataset.setup_docker --task os

    # Only dbbench images:
    conda run -n master python -m generate_dataset.setup_docker --task db

    # Skip already-present images:
    conda run -n master python -m generate_dataset.setup_docker --skip-existing

What it builds / pulls
----------------------
OS task (os-std):
  pull  ubuntu:latest           ← base for local-os images
  pull  mysql:8                 ← needed by dbbench but pulled here too
  build local-os/default        ← data/os_interaction/res/dockerfiles/default
  build local-os/packages       ← data/os_interaction/res/dockerfiles/packages
  build local-os/ubuntu         ← data/os_interaction/res/dockerfiles/ubuntu
  build agentbench/os-worker    ← src/server/tasks/os_interaction/Dockerfile

DB task (dbbench-std):
  pull  mysql:8
  build agentbench/db-worker    ← src/server/tasks/dbbench/Dockerfile

Note on docker.1ms.run/ubuntu
------------------------------
The os_interaction Dockerfiles use `docker.1ms.run/ubuntu` as their base
(a Chinese CDN mirror).  This script patches those Dockerfiles on the fly
to use `ubuntu:latest` if the mirror is unreachable, so the build works
everywhere without editing the original files.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Image definitions
# ---------------------------------------------------------------------------
# Each entry: (kind, tag, context_dir, dockerfile_path_or_content)
# kind = "pull" | "build"

# Images needed to run the os-std task end-to-end (trace generation).
#
# Sandbox image usage, measured from data/os_interaction/data/**.json:
#   "local": "default"   → 55 problems  (also the fallback when no `create` block)
#   "local": "packages"  →  1 problem
#   local-os/ubuntu      → not referenced by the std data, built for parity with
#                          the upstream README (cheap: ubuntu + one echo)
OS_IMAGES = [
    # ── base + infrastructure pulls ──
    {
        "kind": "pull",
        "tag":  "ubuntu:latest",                 # base for the local-os/* images
    },
    {
        "kind": "pull",
        "tag":  "jingbh/agentrl-controller:latest",   # FC controller (~90 MB)
    },
    {
        "kind": "pull",
        "tag":  "redis:7",                       # worker state / sandbox locking (~192 MB)
    },

    # ── sandbox images the agent's bash runs inside ──
    {
        "kind":       "build",
        "tag":        "local-os/default",         # ~970 MB
        "context":    REPO_ROOT / "data/os_interaction/res/dockerfiles",
        "dockerfile": REPO_ROOT / "data/os_interaction/res/dockerfiles/default",
    },
    {
        "kind":       "build",
        "tag":        "local-os/packages",        # ~1.1 GB (adds npm, git, …)
        "context":    REPO_ROOT / "data/os_interaction/res/dockerfiles",
        "dockerfile": REPO_ROOT / "data/os_interaction/res/dockerfiles/packages",
    },
    {
        "kind":       "build",
        "tag":        "local-os/ubuntu",
        "context":    REPO_ROOT / "data/os_interaction/res/dockerfiles",
        "dockerfile": REPO_ROOT / "data/os_interaction/res/dockerfiles/ubuntu",
    },

    # ── the os-std task worker itself ──
    {
        "kind":       "build",
        "tag":        "agentbench/os-worker",     # ~1.8 GB
        "context":    REPO_ROOT,
        "dockerfile": REPO_ROOT / "src/server/tasks/os_interaction/Dockerfile",
    },
]

# Only needed if you also want the dbbench-std task. Not required for the
# os-std trace generation that this pipeline targets.
DB_IMAGES = [
    {
        "kind": "pull",
        "tag":  "mysql:8",                        # ~580 MB
    },
    {
        "kind":       "build",
        "tag":        "agentbench/db-worker",
        "context":    REPO_ROOT,
        "dockerfile": REPO_ROOT / "src/server/tasks/dbbench/Dockerfile",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a shell command, stream output, return True on success."""
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run — skipped]")
        return True
    # The worker Dockerfiles use `RUN --mount=type=cache,...`, which requires
    # BuildKit. Enable it here so a fresh machine doesn't fail with
    # "the --mount option requires BuildKit".
    env = dict(os.environ)
    env["DOCKER_BUILDKIT"] = "1"
    env["COMPOSE_DOCKER_CLI_BUILD"] = "1"
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"\n  ❌  Command failed (exit {result.returncode})")
        return False
    return True


def _image_exists(tag: str) -> bool:
    """Return True if the image tag is already present in local Docker."""
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _mirror_reachable(timeout: int = 5) -> bool:
    """Check if docker.1ms.run is reachable (Chinese CDN mirror)."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo("docker.1ms.run", 443)
        return True
    except Exception:
        return False


def _patched_dockerfile(original: Path, use_mirror: bool) -> Path:
    """
    Return a path to a (possibly patched) Dockerfile.
    If the mirror is unreachable, replaces `docker.1ms.run/ubuntu`
    with `ubuntu:latest` in a temp copy.
    Returns the original path unchanged if no patch is needed.
    """
    content = original.read_text()
    if "docker.1ms.run/ubuntu" not in content:
        return original
    if use_mirror:
        return original

    # Patch: replace mirror with docker hub ubuntu
    patched = content.replace("docker.1ms.run/ubuntu", "ubuntu:latest")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".Dockerfile", delete=False, dir=original.parent
    )
    tmp.write(patched)
    tmp.flush()
    return Path(tmp.name)


def _check_docker() -> None:
    """Raise SystemExit if Docker CLI is missing or daemon is not running."""
    docker_path = shutil.which("docker")
    if docker_path is None:
        print("❌  Docker CLI not found in PATH.")
        print()
        print("  macOS  : Install Docker Desktop → https://docs.docker.com/desktop/install/mac-install/")
        print("  Linux  : sudo apt-get install docker.io  (Ubuntu/Debian)")
        print("           sudo systemctl start docker")
        print()
        print("  On the cluster (no sudo): use rootless Docker or Podman.")
        print("  Podman is Docker-compatible: https://podman.io/docs/installation")
        sys.exit(1)

    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print("❌  Docker daemon is not running.")
        print("    macOS (Colima) : colima start")
        print("    macOS (Desktop): start Docker Desktop")
        print("    Linux          : sudo systemctl start docker")
        sys.exit(1)
    print(f"✅  Docker found at {docker_path} and daemon is running.")

    # BuildKit needs the buildx plugin; the worker Dockerfiles use --mount=cache.
    buildx = subprocess.run(
        ["docker", "buildx", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if buildx.returncode != 0:
        print("❌  `docker buildx` not available, but the worker Dockerfiles need BuildKit.")
        print("    macOS : brew install docker-buildx")
        print("            then add to ~/.docker/config.json:")
        print('              "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]')
        print("    Linux : sudo apt-get install docker-buildx-plugin")
        sys.exit(1)
    print("✅  docker buildx available (BuildKit enabled).")


# ---------------------------------------------------------------------------
# Core build/pull logic
# ---------------------------------------------------------------------------

def process_images(
    images: list[dict],
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict[str, bool]:
    """
    Process a list of image definitions (pull or build).
    Returns {tag: success_bool}.
    """
    # Decide once whether the CDN mirror is reachable
    use_mirror = _mirror_reachable()
    if not use_mirror:
        print("  ℹ️  docker.1ms.run unreachable — Dockerfiles will use ubuntu:latest instead.\n")

    results = {}
    temp_files: list[Path] = []

    for img in images:
        tag  = img["tag"]
        kind = img["kind"]

        print(f"\n{'─' * 55}")
        print(f"  [{kind.upper()}]  {tag}")

        if skip_existing and _image_exists(tag):
            print(f"  ⏭   Image already present — skipping.")
            results[tag] = True
            continue

        if kind == "pull":
            ok = _run(["docker", "pull", tag], dry_run)

        elif kind == "build":
            dockerfile = img["dockerfile"]
            context    = img["context"]

            # Patch dockerfile if mirror unavailable
            patched_df = _patched_dockerfile(dockerfile, use_mirror)
            if patched_df != dockerfile:
                temp_files.append(patched_df)
                print(f"  ℹ️  Using patched Dockerfile (ubuntu mirror replaced)")

            ok = _run(
                [
                    "docker", "build",
                    "--tag",  tag,
                    "--file", str(patched_df),
                    str(context),
                ],
                dry_run,
            )
        else:
            print(f"  ⚠️  Unknown kind '{kind}' — skipping.")
            ok = False

        results[tag] = ok
        if ok and not dry_run:
            print(f"  ✅  {tag}")

    # Clean up any temp Dockerfiles
    for tmp in temp_files:
        try:
            tmp.unlink()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and pull Docker images required by AgentBench tasks."
    )
    parser.add_argument(
        "--task",
        choices=["os", "db", "all"],
        default="all",
        help="Which task images to prepare (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images already present in local Docker.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  AgentBench — Docker image setup")
    print(f"  Task scope : {args.task.upper()}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Skip exist : {args.skip_existing}")
    print("=" * 55)

    if not args.dry_run:
        _check_docker()

    images = []
    if args.task in ("os", "all"):
        images += OS_IMAGES
    if args.task in ("db", "all"):
        # avoid duplicate mysql:8 pull if both tasks selected
        db_images = [i for i in DB_IMAGES if i["tag"] not in {x["tag"] for x in images}]
        images += db_images

    t0 = time.perf_counter()
    results = process_images(images, dry_run=args.dry_run, skip_existing=args.skip_existing)
    elapsed = time.perf_counter() - t0

    # ── Summary ──
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY  ({elapsed:.0f}s)")
    print(f"{'=' * 55}")
    passed  = [t for t, ok in results.items() if ok]
    failed  = [t for t, ok in results.items() if not ok]
    skipped = []

    for tag in passed:
        print(f"  ✅  {tag}")
    for tag in failed:
        print(f"  ❌  {tag}")

    if failed:
        print(f"\n  {len(failed)} image(s) failed. Fix errors above and re-run.")
        print("  Tip: re-run with --skip-existing to skip already-built images.")
        sys.exit(1)
    else:
        print(f"\n  All {len(passed)} images ready.")
        print("\n  Next step — start the task server stack:")
        print("    LOCAL (os only): conda run -n master python -m generate_dataset.serve_local up")
        print("    CLUSTER (all)  : docker compose -f extra/docker-compose.yml up --build")


if __name__ == "__main__":
    main()
