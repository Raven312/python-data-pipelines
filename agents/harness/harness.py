"""Agent Harness — the controlled execution environment for pipeline agents.

The harness is the boundary between a decision-maker (rule-based policy or an
LLM) and the data platform. The agent never touches Spark, storage, or
credentials directly; it can only ask the harness to invoke a registered tool.

    ┌──────────────┐  invoke(name, args)  ┌───────────────────────────────┐
    │ Policy / LLM │ ───────────────────> │          AgentHarness         │
    └──────────────┘                      │ 1. guardrails (perm, budget,  │
           ▲                              │    schema, approval)          │
           │          ToolResult          │ 2. pre_tool hooks             │
           └───────────────────────────── │ 3. handler (the real work)    │
                                          │ 4. post_tool / on_error hooks │
                                          │ 5. audit record               │
                                          └───────────────────────────────┘

`invoke` never raises. Every outcome — success, handler exception, guardrail
block — comes back as a ToolResult so the loop can observe it and decide.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from agents.harness.audit import ActionRecord, AuditTrail
from agents.harness.guardrails import GuardrailPolicy
from agents.harness.hooks import HookManager, ToolCall
from agents.harness.tools import ToolRegistry

logger = logging.getLogger(__name__)

_SUMMARY_LIMIT = 500


@dataclass
class ToolResult:
    """Outcome of a single harness invocation."""

    tool_name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    blocked: bool = False
    latency_ms: float = 0.0

    @property
    def status(self) -> str:
        if self.blocked:
            return "blocked"
        return "ok" if self.ok else "error"


def _summarize(value: Any) -> Any:
    """Keep audit records small: dicts pass through, long reprs are cut."""
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        text = repr(value)
        return value if len(text) <= _SUMMARY_LIMIT else text[:_SUMMARY_LIMIT] + "…"
    return f"<{type(value).__name__}>"


class AgentHarness:
    """Executes tool calls on behalf of an agent, under policy.

    Args:
        registry: The bounded set of tools the agent may use.
        policy: Guardrails applied before each call.
        hooks: Lifecycle hooks; a fresh HookManager if omitted.
        audit: Audit trail; in-memory if omitted.
        session_id: Correlates every record from one agent session.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Optional[GuardrailPolicy] = None,
        hooks: Optional[HookManager] = None,
        audit: Optional[AuditTrail] = None,
        session_id: Optional[str] = None,
    ):
        self.registry = registry
        self.policy = policy or GuardrailPolicy()
        self.hooks = hooks or HookManager()
        self.audit = audit or AuditTrail()
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.call_counts: dict[str, int] = {}
        self._step = 0

    def invoke(self, tool_name: str, args: Optional[dict] = None) -> ToolResult:
        args = dict(args or {})
        self._step += 1
        call = ToolCall(tool_name, args, self._step, self.session_id)
        tool = self.registry.get(tool_name)

        verdict = self.policy.evaluate(tool, tool_name, args, self.call_counts)
        if not verdict.allowed:
            return self._finish(call, ToolResult(tool_name, False, error=verdict.reason, blocked=True))

        hook_block = self.hooks.run_pre(call)
        if hook_block:
            return self._finish(
                call, ToolResult(tool_name, False, error=f"blocked by hook: {hook_block}", blocked=True)
            )

        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        start = time.monotonic()
        try:
            output = tool.handler(**args)
            result = ToolResult(tool_name, True, output=output)
        except Exception as exc:  # the agent must observe failures, not crash on them
            logger.warning("[HARNESS] %s raised %s: %s", tool_name, type(exc).__name__, exc)
            self.hooks.run_error(call, exc)
            result = ToolResult(tool_name, False, error=f"{type(exc).__name__}: {exc}")
        result.latency_ms = round((time.monotonic() - start) * 1000, 2)

        return self._finish(call, result)

    def _finish(self, call: ToolCall, result: ToolResult) -> ToolResult:
        tool = self.registry.get(call.tool_name)
        self.audit.record(ActionRecord(
            session_id=self.session_id,
            step=call.step,
            tool_name=call.tool_name,
            args=call.args,
            status=result.status,
            permission=tool.permission.value if tool else None,
            latency_ms=result.latency_ms,
            output_summary=_summarize(result.output),
            error=result.error,
        ))
        if not result.blocked:
            self.hooks.run_post(call, result)

        log_fn = logger.info if result.ok else logger.warning
        log_fn(
            "[HARNESS] step=%d %s → %s%s",
            call.step, call.tool_name, result.status,
            f" ({result.error})" if result.error else "",
        )
        return result

    def tool_specs(self) -> list[dict]:
        """Tool specs within the policy ceiling, for an LLM planner."""
        return self.registry.specs(self.policy.max_permission)
