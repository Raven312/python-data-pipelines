"""Claude policy — an LLM planner for the agent loop.

Drop-in replacement for RuleBasedPolicy. Claude sees the harness's tool specs
(already filtered by the guardrail ceiling) plus a `finish` tool, and picks
one tool per turn. Every pick still goes through the harness, so the model
cannot exceed its permissions, budgets, or approval gates — the harness is
what makes it safe to let a model operate the pipeline.

Requires the optional `anthropic` package and ANTHROPIC_API_KEY:

    pip install anthropic
    policy = ClaudePolicy(harness.tool_specs())
"""

import json
from typing import Any, Optional

from agents.loop.agent_loop import Action, Decision, Finish, LoopStatus, Observation

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You operate a PySpark Medallion ETL pipeline (Bronze → Silver → Gold) \
through tools. Work one tool call at a time.

Operating rules:
- Start with inspect_source_schema. If it returns "reject", escalate.
- Run stages in order: run_bronze, run_bronze_gate, run_silver, run_silver_gate, \
profile_silver_quality, run_gold, publish_outputs.
- On a transient error (throttling, 503, timeout) retry the same stage, at most 3 times.
- On a data quality failure, quarantine_batch for that stage, then escalate with a \
root-cause summary a data engineer can act on.
- Never retry a blocked call; blocked means a guardrail or a human said no.
- Call finish with status "succeeded" once publish_outputs succeeds, or "escalated" \
after escalate succeeds."""

FINISH_TOOL = {
    "name": "finish",
    "description": "End the run. Use 'succeeded' only after publish_outputs succeeded.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["succeeded", "escalated", "blocked"]},
            "reason": {"type": "string"},
        },
        "required": ["status", "reason"],
    },
}


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str)[:4000]


class ClaudePolicy:
    """Plans the next pipeline action with Claude's tool use.

    Args:
        tool_specs: Model-facing specs, normally `harness.tool_specs()`.
        model: Claude model id.
        client: An `anthropic.Anthropic`-compatible client (injected in tests).
        max_tokens: Per-turn output cap.
    """

    def __init__(
        self,
        tool_specs: list[dict],
        model: str = DEFAULT_MODEL,
        client: Optional[Any] = None,
        max_tokens: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if client is None:
            import anthropic  # optional dependency, imported only when used

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.tools = list(tool_specs) + [FINISH_TOOL]
        self.messages: list[dict] = []
        self._pending_tool_use_id: Optional[str] = None

    def decide(self, obs: Observation) -> Decision:
        if not self.messages:
            self.messages.append({"role": "user", "content": f"Goal: {obs.goal}"})
        elif self._pending_tool_use_id and obs.last:
            result = obs.last.result
            payload = result.output if result.ok else {"error": result.error, "blocked": result.blocked}
            self.messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": self._pending_tool_use_id,
                    "content": _to_json(payload),
                    "is_error": not result.ok,
                }],
            })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            tools=self.tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})

        rationale = " ".join(b.text for b in response.content if b.type == "text").strip()
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            return Finish(LoopStatus.FAILED, "model returned no tool call")

        if tool_use.name == "finish":
            self._pending_tool_use_id = None
            status = {
                "succeeded": LoopStatus.SUCCEEDED,
                "escalated": LoopStatus.ESCALATED,
                "blocked": LoopStatus.BLOCKED,
            }.get(tool_use.input.get("status"), LoopStatus.FAILED)
            return Finish(status, tool_use.input.get("reason", ""))

        self._pending_tool_use_id = tool_use.id
        return Action(tool_use.name, dict(tool_use.input), rationale=rationale[:200])
