"""Guardrails — the policy layer that decides whether a tool call may run.

Checks run before any handler executes, in this order:

    1. Tool exists in the registry            (no hallucinated capabilities)
    2. Permission tier is within the ceiling  (e.g. a monitor is READ-only)
    3. Per-tool and total call budgets        (no runaway loops)
    4. Arguments match the tool's schema      (reject malformed model output)
    5. Approval gate for gated tiers          (human-in-the-loop before writes)

A denied call never raises; it returns a GuardrailVerdict the harness turns
into a blocked ToolResult, so the agent observes the refusal and can adapt.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.harness.tools import Permission, Tool

# Approver receives (tool_name, args) and returns True to allow the call.
Approver = Callable[[str, dict], bool]


def deny_all(tool_name: str, args: dict) -> bool:
    """Default approver: nothing gated runs without an explicit human hook."""
    return False


def approve_all(tool_name: str, args: dict) -> bool:
    """Approver for tests and local demos where no human is in the loop."""
    return True


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: str = "OK"


@dataclass
class GuardrailPolicy:
    """Declarative limits on what an agent may do through the harness.

    Attributes:
        max_permission: Highest tier the agent may call at all.
        require_approval: Tiers that need the approver's consent per call.
        max_calls_per_tool: Hard cap on invocations of any single tool.
        max_total_calls: Hard cap on all invocations in one session.
        denylist: Tool names blocked regardless of tier.
        approver: Callback deciding gated calls (defaults to deny).
    """

    max_permission: Permission = Permission.WRITE
    require_approval: set[Permission] = field(
        default_factory=lambda: {Permission.DESTRUCTIVE}
    )
    max_calls_per_tool: int = 10
    max_total_calls: int = 50
    denylist: set[str] = field(default_factory=set)
    approver: Approver = deny_all

    def evaluate(
        self,
        tool: Optional[Tool],
        tool_name: str,
        args: dict,
        call_counts: dict[str, int],
    ) -> GuardrailVerdict:
        if tool is None:
            return GuardrailVerdict(False, f"unknown tool '{tool_name}'")

        if tool_name in self.denylist:
            return GuardrailVerdict(False, f"tool '{tool_name}' is denylisted")

        if tool.permission.rank > self.max_permission.rank:
            return GuardrailVerdict(
                False,
                f"'{tool_name}' needs {tool.permission.value} permission; "
                f"ceiling is {self.max_permission.value}",
            )

        total = sum(call_counts.values())
        if total >= self.max_total_calls:
            return GuardrailVerdict(
                False, f"total call budget exhausted ({self.max_total_calls})"
            )

        if call_counts.get(tool_name, 0) >= self.max_calls_per_tool:
            return GuardrailVerdict(
                False,
                f"call budget for '{tool_name}' exhausted ({self.max_calls_per_tool})",
            )

        schema_error = tool.validate(args)
        if schema_error:
            return GuardrailVerdict(False, f"invalid arguments: {schema_error}")

        if tool.permission in self.require_approval and not self.approver(tool_name, args):
            return GuardrailVerdict(
                False, f"approval denied for {tool.permission.value} tool '{tool_name}'"
            )

        return GuardrailVerdict(True)

    @classmethod
    def read_only(cls) -> "GuardrailPolicy":
        """Preset for monitoring agents: observe, never change anything."""
        return cls(max_permission=Permission.READ, require_approval=set())

    @classmethod
    def operator(cls, approver: Approver = deny_all) -> "GuardrailPolicy":
        """Preset for a pipeline operator: may run stages; publishing is gated."""
        return cls(
            max_permission=Permission.DESTRUCTIVE,
            require_approval={Permission.DESTRUCTIVE},
            approver=approver,
        )
