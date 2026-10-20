"""
generate_dataset/patch.py
=========================
Monkey-patches two AgentBench classes so that token/tool-call metrics are
captured during a run WITHOUT touching any framework source files.

    apply_patches()   ← call this once before starting the Assigner

After patching:
  • HTTPAgent.inference() stores the last vLLM `usage` dict in
    `agent.last_usage` after every forward call.
  • OSInteraction.start_sample() accumulates per-turn usage into a
    `_usage_accumulator` on the session, then attaches a final `"usage"`
    key to result.result before returning.
  • OSInteraction.calculate_overall() computes aggregate efficiency stats
    alongside the existing pass/fail numbers.

The patches are additive — all original behaviour is preserved.
"""

import json
import logging
import time
from statistics import mean
from typing import List

from config import USAGE_KEY_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1 — HTTPAgent: capture vLLM usage after every inference call
# ---------------------------------------------------------------------------

def _patch_http_agent():
    from src.client.agents.http_agent import HTTPAgent

    original_inference = HTTPAgent.inference

    def patched_inference(self, history: List[dict]) -> str:
        """Wraps original inference; stores usage dict from vLLM response."""
        # We re-implement the retry loop so we can capture the raw JSON
        # before it is formatted through return_format.
        import requests
        import time as _time
        from src.client.agents.http_agent import no_ssl_verification
        from src.typings import AgentContextLimitException, AgentClientException
        from src.client.agents.http_agent import check_context_limit

        self.last_usage = {}   # reset before every call

        for attempt in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))
                with no_ssl_verification():
                    resp = requests.post(
                        self.url,
                        json=body,
                        headers=self.headers,
                        proxies=self.proxies,
                        timeout=120,
                    )
                if resp.status_code != 200:
                    if check_context_limit(resp.text):
                        raise AgentContextLimitException(resp.text)
                    else:
                        raise Exception(
                            f"Invalid status code {resp.status_code}:\n\n{resp.text}"
                        )
            except AgentClientException:
                raise
            except Exception as exc:
                logger.warning("HTTPAgent inference attempt %d failed: %s", attempt + 1, exc)
            else:
                resp_json = resp.json()
                # Capture usage — vLLM always includes this field
                self.last_usage = resp_json.get("usage", {})
                return self.return_format.format(response=resp_json)
            _time.sleep(attempt + 2)

        raise Exception("HTTPAgent.inference failed after 3 attempts.")

    HTTPAgent.inference = patched_inference
    # Ensure every instance starts with the attribute even if inference is
    # called before the first successful response.
    original_init = HTTPAgent.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.last_usage = {}

    HTTPAgent.__init__ = patched_init
    logger.info("Patched HTTPAgent.inference to capture vLLM usage.")


# ---------------------------------------------------------------------------
# Patch 2 — OSInteraction: accumulate usage per turn + enrich result
# ---------------------------------------------------------------------------

def _patch_os_interaction():
    from src.server.tasks.os_interaction.task import OSInteraction
    from agentrl.worker.task import Session

    # ---- 2a. patch _handle_agent_action to record usage after each turn ----

    original_handle = OSInteraction._handle_agent_action

    async def patched_handle_agent_action(
        self, session: Session, container, round_num, finish,
        round_reward, function_name, call_id
    ) -> dict:
        result = await original_handle(
            self, session, container, round_num, finish,
            round_reward, function_name, call_id
        )

        # Pull usage from the agent attached to the session.
        # AgentRL's Session keeps a reference to the agent that submitted
        # the last response; we access it via session._agent if present.
        agent = getattr(session, "_agent", None)
        if agent is None:
            # Fallback: look for the most recently created HTTPAgent instance
            # stored on the task itself (set in start_sample patch below).
            agent = getattr(self, "_current_agent", None)

        if agent is not None:
            usage = getattr(agent, "last_usage", {})
            acc = getattr(session, "_usage_accumulator", None)
            if acc is not None and usage:
                for vllm_key, friendly_key in USAGE_KEY_MAP.items():
                    acc[friendly_key] = acc.get(friendly_key, 0) + usage.get(vllm_key, 0)
                acc["tool_calls"] += 1 if result.get("function_name") else 0
                acc["turns"] += 1

        return result

    OSInteraction._handle_agent_action = patched_handle_agent_action

    # ---- 2b. patch start_sample to init accumulator + attach to result ----

    original_start_sample = OSInteraction.start_sample

    async def patched_start_sample(self, index, session: Session):
        # Initialise the per-session accumulator
        session._usage_accumulator = {
            "input_tokens":  0,
            "output_tokens": 0,
            "total_tokens":  0,
            "tool_calls":    0,
            "turns":         0,
        }

        result = await original_start_sample(self, index, session)

        # Attach accumulated usage to result.result so it flows through
        # TaskOutput → runs.jsonl → calculate_overall
        if result.result is None:
            result.result = {}
        acc = getattr(session, "_usage_accumulator", {})
        result.result["usage"] = acc

        return result

    OSInteraction.start_sample = patched_start_sample

    # ---- 2c. patch calculate_overall to add efficiency block ----

    original_calculate_overall = OSInteraction.calculate_overall

    def patched_calculate_overall(self, results) -> dict:
        overall = original_calculate_overall(self, results)

        # Collect usage dicts from all completed results
        usages = [
            r.result.get("usage", {})
            for r in results
            if r and r.result and isinstance(r.result.get("usage"), dict)
        ]

        if usages:
            def _avg(key):
                vals = [u.get(key, 0) for u in usages]
                return round(mean(vals), 2) if vals else 0.0

            overall["efficiency"] = {
                "avg_input_tokens":  _avg("input_tokens"),
                "avg_output_tokens": _avg("output_tokens"),
                "avg_total_tokens":  _avg("total_tokens"),
                "avg_tool_calls":    _avg("tool_calls"),
                "avg_turns":         _avg("turns"),
                # Per-sample raw numbers for later analysis
                "samples": [
                    {
                        "index":         getattr(r, "index", None),
                        "pass":          bool(r.result.get("result", False)) if r.result else False,
                        **r.result.get("usage", {}),
                    }
                    for r in results if r and r.result
                ],
            }
        else:
            logger.warning(
                "No usage data found in results — "
                "vLLM may not have returned usage fields."
            )
            overall["efficiency"] = {}

        return overall

    OSInteraction.calculate_overall = patched_calculate_overall
    logger.info("Patched OSInteraction to accumulate and report efficiency metrics.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_patches():
    """
    Apply all patches.  Call this once at the very start of run.py,
    before any AgentBench classes are instantiated.

    Two execution contexts:
      • HOST (assigner side): only HTTPAgent exists here.  `agentrl` is NOT
        installed on the host (it lives inside the task-worker Docker image),
        so the OSInteraction patch is skipped gracefully.
      • WORKER (inside the task container): `agentrl` IS importable, so the
        OSInteraction patch applies and captures per-turn token usage.

    The HTTPAgent patch is what the host needs for token capture; the
    OSInteraction patch only matters where the task actually runs.
    """
    _patch_http_agent()
    try:
        _patch_os_interaction()
    except ModuleNotFoundError as exc:
        # agentrl not present → we are on the host (assigner side). Expected.
        logger.info(
            "Skipping OSInteraction patch (agentrl not on this host — "
            "expected on the assigner side): %s", exc
        )
    logger.info("generate_dataset patches applied.")
