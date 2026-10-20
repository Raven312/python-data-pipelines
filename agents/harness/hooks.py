"""Lifecycle hooks — extension points around every tool call.

Hooks let you attach cross-cutting behaviour (alerting, metrics emission,
redaction, extra policy) without touching tool handlers or the loop.

    pre_tool(call)          → may return a string to block the call
    post_tool(call, result) → observe the outcome (e.g. emit metrics)
    on_error(call, exc)     → observe handler exceptions

Hook failures are logged and swallowed: an observability bug must never take
the pipeline down.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from agents.harness.harness import ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """An attempted invocation, as seen by hooks."""

    tool_name: str
    args: dict
    step: int
    session_id: str


PreHook = Callable[[ToolCall], Optional[str]]
PostHook = Callable[[ToolCall, "ToolResult"], None]
ErrorHook = Callable[[ToolCall, Exception], None]


class HookManager:
    """Registry and dispatcher for harness lifecycle hooks."""

    def __init__(self) -> None:
        self._pre: list[PreHook] = []
        self._post: list[PostHook] = []
        self._error: list[ErrorHook] = []

    def pre_tool(self, fn: PreHook) -> PreHook:
        self._pre.append(fn)
        return fn

    def post_tool(self, fn: PostHook) -> PostHook:
        self._post.append(fn)
        return fn

    def on_error(self, fn: ErrorHook) -> ErrorHook:
        self._error.append(fn)
        return fn

    def run_pre(self, call: ToolCall) -> Optional[str]:
        """Run pre-hooks in order; the first non-None return blocks the call."""
        for hook in self._pre:
            try:
                reason = hook(call)
            except Exception:
                logger.exception("[HOOK] pre_tool hook failed for %s", call.tool_name)
                continue
            if reason:
                return reason
        return None

    def run_post(self, call: ToolCall, result: "ToolResult") -> None:
        for hook in self._post:
            try:
                hook(call, result)
            except Exception:
                logger.exception("[HOOK] post_tool hook failed for %s", call.tool_name)

    def run_error(self, call: ToolCall, exc: Exception) -> None:
        for hook in self._error:
            try:
                hook(call, exc)
            except Exception:
                logger.exception("[HOOK] on_error hook failed for %s", call.tool_name)
