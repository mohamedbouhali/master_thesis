"""
generate_dataset/enroot_backend.py
==================================
Drop-in replacement for `agent_loop.ControllerClient` that runs the os-std task
WITHOUT Docker, controller, redis or worker. Each sample gets an Enroot
container (rootless, fake root) built from a prepared squashfs image.

Same interface as ControllerClient: list_indices, worker_capacity,
start_sample, interact, cancel. So `agent_loop.run_sample` is unchanged.

Setup once:  bash generate_dataset/setup_enroot.sh
Images live in $ENROOT_SANDBOX_DIR (default ~/enroot_images): default.sqsh, packages.sqsh, ubuntu.sqsh
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = Path(os.environ.get("ENROOT_SANDBOX_DIR", Path.home() / "enroot_images"))
OS_CFG = REPO / "configs/tasks/os.yaml"

SYSTEM_PROMPT = """You are an assistant that will act like a person. I will play the role of a Linux (Ubuntu) operating system.
Your goal is to implement the operations required by me or answer the questions proposed by me.
For each of your turns, you should first think about what you should do, and then call exactly one of the provided tools according to the situation.
If you think the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself.
Attention, your bash code should not contain any input operation. Once again, you should use one tool in each turn, and should not respond without function calling.
Note that if you think the task has been finished, or there is some message missing to completely complete the task, you should respond with calling the function "finish_action", as no additional information will be provided.
Also, note that if you have gotten the answer to the question, you should call the "answer_action" tool instead of simply writing your answer in your response.
Your answers should be exact and precise (for example, a single number), do not answer with full sentences or phrases.
Always use a tool provided instead of simply responding with content."""


# ── problem loading (mirrors OSInteraction._load_configs) ─────────────────────
def _script(obj, root):
    if obj is None:
        return None
    if isinstance(obj, str):
        return "bash", obj
    lang = obj.get("language", "bash")
    if "file" in obj:
        return lang, (Path(root) / obj["file"]).read_text(encoding="utf-8")
    return lang, obj["code"]


def _scripts(obj, root):
    if obj is None:
        return []
    return [_script(o, root) for o in (obj if isinstance(obj, list) else [obj])]


def load_problems() -> dict[int, dict]:
    cfg = yaml.safe_load(OS_CFG.read_text())["os-std"]["parameters"]["data_config"]["files"]
    problems, idx = {}, 0
    for item in cfg:
        for f in sorted(glob.glob(str(REPO / item["problem_file"]))):
            raw = json.load(open(f, encoding="utf-8")) if f.endswith(".json") else \
                [json.loads(l) for l in open(f, encoding="utf-8")]
            for it in (raw if isinstance(raw, list) else [raw]):
                ev, create = it["evaluation"], it.get("create", {})
                if not isinstance(create, dict):   # original loader ignores non-dict `create`
                    create = {}
                root = REPO / item["script_dir"]
                problems[idx] = {
                    "description": it["description"],
                    "image": create.get("local", "default"),
                    "init": _scripts(create.get("init"), root),
                    "start": _script(it.get("start"), root),
                    "match": ({"answer": ev["match"], "strip": True} if isinstance(ev.get("match"), str)
                              else ev.get("match")),
                    "check": _scripts(ev.get("check"), root) if "check" in ev else None,
                    "example": _script(ev.get("example"), root),
                }
                idx += 1
    return problems


# ── sandbox ───────────────────────────────────────────────────────────────────
RC = ["--mount", f"{IMG_DIR}/rc.sh:/etc/rc"]  # clean rc (image's own /etc/rc is broken)


def _enroot(*args, **kw):
    return subprocess.run(["enroot", *args], capture_output=True, **kw)


class Sandbox:
    """One Enroot container + one persistent bash (cwd/env survive between commands)."""

    def __init__(self, image: str):
        self.name = f"os_{uuid.uuid4().hex[:10]}"
        sq = IMG_DIR / f"{image}.sqsh"
        if not sq.exists():
            sq = IMG_DIR / "default.sqsh"
        r = _enroot("create", "-n", self.name, str(sq))
        if r.returncode:
            raise RuntimeError(f"enroot create failed: {r.stderr.decode()[:300]}")
        self.proc = subprocess.Popen(
            ["enroot", "start", "--rw", "--root", *RC, self.name, "bash"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.run_shell("cd /root", 30)

    def run_shell(self, cmd: str, timeout: float = 60.0) -> str:
        tag = f"__END_{uuid.uuid4().hex[:8]}__"
        self.proc.stdin.write(f"{cmd}\necho {tag}\n".encode())
        self.proc.stdin.flush()
        fd, buf, end = self.proc.stdout.fileno(), b"", time.time() + timeout
        while tag.encode() not in buf:
            left = end - time.time()
            if left <= 0:
                buf += b"\n[command timed out]"
                break
            if select.select([fd], [], [], left)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
        return buf.decode("utf-8", "replace").split(tag)[0]

    def run_independent(self, script: tuple[str, str], *params) -> tuple[int, str]:
        lang, code = script
        cmd = ["bash", "-c", code, *(["--", *params] if params else [])] if lang == "bash" \
            else ["python3", "-c", code, *params]
        r = _enroot("start", "--rw", "--root", *RC, self.name, *cmd, timeout=300)
        return r.returncode, r.stdout.decode("utf-8", "replace")

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        _enroot("remove", "-f", self.name)


# ── controller-compatible client ──────────────────────────────────────────────
class EnrootController:
    def __init__(self, *_, round_limit: int = 8, **__):
        self.problems = load_problems()
        self.tools = yaml.safe_load(OS_CFG.read_text())["default"]["parameters"]["tools"]
        self.round_limit = round_limit
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.ids = itertools.count(1)

    def list_indices(self, task):
        return list(self.problems)

    def worker_capacity(self, task):
        return 10_000

    def start_sample(self, task, index):
        p = self.problems[int(index)]
        sb = Sandbox(p["image"])
        try:
            for s in p["init"]:
                code, out = sb.run_independent(s)
                if code:
                    raise RuntimeError(f"init script failed: {out[:300]}")
            if p["start"]:
                sb.run_shell(p["start"][1], 120)
        except Exception:
            sb.close()
            raise
        sid = str(next(self.ids))
        with self.lock:
            self.sessions[sid] = {"sb": sb, "p": p, "round": 0}
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                 f"Now, I will start a new problem in a new OS. My problem is:\n\n{p['description']}"}]
        return sid, msgs, self.tools

    def interact(self, sid, msg):
        s = self.sessions[sid]
        s["round"] += 1
        calls = msg.get("tool_calls") or []
        if not calls:
            return self._step(False, "running",
                              [{"role": "user", "content": "No executable tool calls found. Please call a tool instead"}],
                              s)
        call = calls[0]
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception as e:
            return self._step(False, "running",
                              [{"role": "tool", "content": str(e), "tool_call_id": call["id"]}], s)
        if name == "bash_action":
            out = s["sb"].run_shell(str(args.get("script", "")))
            if len(out) > 800:
                out = out[:780] + "\n[truncated because the output is too long]"
            text = f"The output of the OS:\n\n{out}" if out.strip() else "The output of the OS is empty."
            return self._step(False, "running",
                              [{"role": "tool", "content": text, "tool_call_id": call["id"]}], s)
        if name in ("answer_action", "finish_action"):
            answer = args.get("answer") if name == "answer_action" else None
            ok = self._evaluate(s, answer)
            self._close(sid)
            return {"finish": True, "status": "completed", "reward": int(ok),
                    "metrics": {"score": int(ok)}, "messages": []}
        return self._step(False, "running",
                          [{"role": "tool", "content": "Invalid function call. Please call a tool instead",
                            "tool_call_id": call["id"]}], s)

    def _step(self, finish, status, messages, s):
        if s["round"] >= self.round_limit:
            sid = next(k for k, v in self.sessions.items() if v is s)
            self._close(sid)
            return {"finish": True, "status": "task_limit_reached", "reward": 0,
                    "metrics": {"score": 0}, "messages": messages}
        return {"finish": finish, "status": status, "reward": 0,
                "metrics": {"score": 0}, "messages": messages}

    def _evaluate(self, s, answer) -> bool:
        p = s["p"]
        if p["match"]:
            a = answer.strip() if isinstance(answer, str) and p["match"].get("strip") else answer
            if "answer" in p["match"]:
                return a == p["match"]["answer"]
            if "regex" in p["match"]:
                import re
                return re.search(p["match"]["regex"], a or "") is not None
            return False
        params = [str(answer)]
        for script in p["check"] or []:
            code, out = s["sb"].run_independent(script or p["example"], *params)
            if code:
                return False
            params.append(out)
        return True

    def _close(self, sid):
        with self.lock:
            s = self.sessions.pop(sid, None)
        if s:
            s["sb"].close()

    def cancel(self, sid):
        self._close(sid)
