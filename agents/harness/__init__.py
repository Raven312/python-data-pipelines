"""Agent Harness — the bounded, audited execution layer for pipeline agents.

An agent never calls Spark or storage directly. It asks the harness to invoke
a registered Tool; the harness applies guardrails (permission tier, budgets,
schema validation, approval gate), runs lifecycle hooks, executes the
handler, and writes an audit record. See agents/harness/harness.py.
"""

from agents.harness.audit import ActionRecord, AuditTrail
from agents.harness.guardrails import GuardrailPolicy, GuardrailVerdict, approve_all, deny_all
from agents.harness.harness import AgentHarness, ToolResult
from agents.harness.hooks import HookManager, ToolCall
from agents.harness.tools import Permission, Tool, ToolRegistry

__all__ = [
    "ActionRecord",
    "AgentHarness",
    "AuditTrail",
    "GuardrailPolicy",
    "GuardrailVerdict",
    "HookManager",
    "Permission",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "approve_all",
    "deny_all",
]
