"""Agent Loop — the observe → decide → act cycle that drives the pipeline.

    ┌─────────────────────────────────────────────────────────────────┐
    │                           AgentLoop                             │
    │                                                                 │
    │   ┌──────────┐   Observation   ┌────────┐  Action  ┌─────────┐   │
    │   │ OBSERVE  │ ──────────────> │ DECIDE │ ───────> │   ACT   │   │
    │   │ history, │                 │ Policy │          │ Harness │   │
    │   │ results  │ <────────────── │        │          │ .invoke │   │
    │   └──────────┘   ToolResult    └────────┘          └─────────┘   │
    │         ▲                          │ Finish                  │   │
    │         └──────────────────────────┼─────────────────────────┘   │
    │                                    ▼                             │
    │   Stop when: policy finishes · max_steps · time budget ·         │
    │              too many consecutive failures                       │
    └─────────────────────────────────────────────────────────────────┘

The loop owns *control flow and limits*; the policy owns *judgment*; the
harness owns *safety*. Swapping the rule-based policy for an LLM changes who
decides, not what the agent is allowed to do or when it must stop.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, Union

from agents.harness.harness import AgentHarness, ToolResult

logger = logging.getLogger(__name__)


class LoopStatus(Enum):
    SUCCEEDED = "succeeded"
    ESCALATED = "escalated"          # agent handed off to a human on purpose
    BLOCKED = "blocked"              # guardrails refused a required action
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"                # gave up after repeated failures


@dataclass
class Action:
    """Policy output: invoke a tool, optionally after a backoff wait."""

    tool: str
    args: dict = field(default_factory=dict)
    rationale: str = ""
    wait_seconds: float = 0.0


@dataclass
class Finish:
    """Policy output: stop the loop with a terminal status."""

    status: LoopStatus
    reason: str


Decision = Union[Action, Finish]


@dataclass
class Step:
    index: int
    action: Action
    result: ToolResult


@dataclass
class Observation:
    """Everything the policy may look at when deciding the next action."""

    goal: str
    step: int
    history: list[Step]
    tools: list[str]

    @property
    def last(self) -> Optional[Step]:
        return self.history[-1] if self.history else None

    def succeeded(self, tool: str) -> bool:
        return any(s.action.tool == tool and s.result.ok for s in self.history)

    def last_result_of(self, tool: str) -> Optional[ToolResult]:
        for s in reversed(self.history):
            if s.action.tool == tool:
                return s.result
        return None


class Policy(Protocol):
    def decide(self, observation: Observation) -> Decision: ...


@dataclass
class LoopResult:
    status: LoopStatus
    reason: str
    steps: list[Step]
    duration_seconds: float
    audit: dict

    @property
    def succeeded(self) -> bool:
        return self.status == LoopStatus.SUCCEEDED

    def transcript(self) -> list[str]:
        lines = []
        for s in self.steps:
            outcome = s.result.status
            detail = f" — {s.result.error}" if s.result.error else ""
            why = f"  [{s.action.rationale}]" if s.action.rationale else ""
            lines.append(f"{s.index:>2}. {s.action.tool:<24} {outcome}{detail}{why}")
        lines.append(f"=> {self.status.value}: {self.reason}")
        return lines

    def incident_report(self) -> dict:
        """Incident-replay summary: what broke, when it was caught, what was done."""
        failures = [s for s in self.steps if not s.result.ok]
        first = failures[0] if failures else None
        return {
            "outcome": self.status.value,
            "reason": self.reason,
            "steps": len(self.steps),
            "incident_detected": first is not None,
            "detected_at_step": first.index if first else None,
            "failed_tool": first.action.tool if first else None,
            "error": first.result.error if first else None,
            "recovery_actions": [
                s.action.tool for s in self.steps
                if first and s.index > first.index
            ],
            "duration_seconds": self.duration_seconds,
        }


class AgentLoop:
    """Runs a policy against a harness until a stop condition is met.

    Args:
        harness: Executes actions under guardrails.
        policy: Decides the next action from the observation.
        max_steps: Hard cap on actions per run.
        max_consecutive_failures: Give up after this many failed actions in a row.
        time_budget_seconds: Wall-clock cap for one run (None = unlimited).
        verify: Checks a claimed success against the actual history and
            returns a reason if it does not hold. The loop trusts results,
            not the policy's word — essential when the policy is an LLM.
        sleep: Injected for tests so backoff waits cost nothing.
        clock: Injected monotonic clock.
    """

    def __init__(
        self,
        harness: AgentHarness,
        policy: Policy,
        max_steps: int = 25,
        max_consecutive_failures: int = 5,
        time_budget_seconds: Optional[float] = None,
        verify: Optional[Callable[[list[Step]], Optional[str]]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.harness = harness
        self.policy = policy
        self.max_steps = max_steps
        self.max_consecutive_failures = max_consecutive_failures
        self.time_budget_seconds = time_budget_seconds
        self.verify = verify
        self.sleep = sleep
        self.clock = clock

    def run(self, goal: str) -> LoopResult:
        start = self.clock()
        history: list[Step] = []
        consecutive_failures = 0
        tools = self.harness.registry.names()

        logger.info("[LOOP] goal: %s", goal)
        while True:
            elapsed = self.clock() - start
            if len(history) >= self.max_steps:
                outcome = Finish(LoopStatus.BUDGET_EXHAUSTED, f"step budget reached ({self.max_steps})")
                break
            if self.time_budget_seconds is not None and elapsed >= self.time_budget_seconds:
                outcome = Finish(LoopStatus.BUDGET_EXHAUSTED, f"time budget reached ({self.time_budget_seconds}s)")
                break
            if consecutive_failures >= self.max_consecutive_failures:
                outcome = Finish(LoopStatus.FAILED, f"{consecutive_failures} consecutive failed actions")
                break

            observation = Observation(goal, len(history) + 1, list(history), tools)
            decision = self.policy.decide(observation)
            if isinstance(decision, Finish):
                outcome = decision
                break

            if decision.wait_seconds > 0:
                logger.info("[LOOP] backing off %.1fs before %s", decision.wait_seconds, decision.tool)
                self.sleep(decision.wait_seconds)

            result = self.harness.invoke(decision.tool, decision.args)
            history.append(Step(len(history) + 1, decision, result))
            consecutive_failures = 0 if result.ok else consecutive_failures + 1

        if outcome.status == LoopStatus.SUCCEEDED and self.verify:
            problem = self.verify(history)
            if problem:
                outcome = Finish(LoopStatus.FAILED, f"success claim rejected: {problem}")

        duration = round(self.clock() - start, 3)
        logger.info("[LOOP] finished: %s — %s (%d steps)", outcome.status.value, outcome.reason, len(history))
        return LoopResult(outcome.status, outcome.reason, history, duration, self.harness.audit.summary())
