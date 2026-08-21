"""Pipeline Orchestration Agent — autonomous pipeline monitor and healer.

This agent watches pipeline executions and takes autonomous action:
  - Retries failed stages with exponential backoff
  - Skips non-critical stages when dependencies fail
  - Sends alerts on persistent failures
  - Tracks pipeline state across runs
  - Makes restart vs. resume decisions based on failure type

Decision loop:
  1. Observe: Poll for pipeline events (success, failure, timeout)
  2. Orient: Classify failure type (transient, data, infrastructure)
  3. Decide: Choose action (retry, skip, alert, escalate)
  4. Act: Execute the chosen action
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class FailureType(Enum):
    TRANSIENT = "transient"
    DATA_QUALITY = "data_quality"
    INFRASTRUCTURE = "infrastructure"
    SCHEMA_DRIFT = "schema_drift"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class AgentAction(Enum):
    RETRY = "retry"
    SKIP = "skip"
    ALERT = "alert"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"
    RESTART = "restart"


@dataclass
class PipelineEvent:
    """An event emitted by a pipeline stage."""

    stage: str
    status: str  # "success", "failure", "timeout"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentDecision:
    """A decision made by the agent in response to an event."""

    event: PipelineEvent
    failure_type: FailureType
    action: AgentAction
    reason: str
    retry_count: int = 0
    max_retries: int = 3


class PipelineOrchestrationAgent:
    """Autonomous agent that monitors and heals pipeline executions.

    The agent follows an OODA (Observe-Orient-Decide-Act) loop to
    autonomously handle pipeline failures without human intervention.
    """

    def __init__(
        self,
        pipeline_name: str,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        alert_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.pipeline_name = pipeline_name
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.alert_callback = alert_callback

        self.retry_counts: dict[str, int] = {}
        self.event_history: list[PipelineEvent] = []
        self.decision_log: list[AgentDecision] = []

    def observe(self, event: PipelineEvent) -> None:
        """Record an incoming pipeline event."""
        self.event_history.append(event)
        logger.info(
            "[OBSERVE] %s.%s → %s", self.pipeline_name, event.stage, event.status
        )

    def orient(self, event: PipelineEvent) -> FailureType:
        """Classify the failure type based on error signals."""
        if event.status == "success":
            return FailureType.TRANSIENT  # not a real failure

        msg = (event.error_message or "").lower()

        if any(kw in msg for kw in ["timeout", "timed out", "deadline"]):
            return FailureType.TIMEOUT

        if any(kw in msg for kw in ["schema", "column", "type mismatch", "structtype"]):
            return FailureType.SCHEMA_DRIFT

        if any(kw in msg for kw in ["quality", "validation", "expectation", "null", "duplicate"]):
            return FailureType.DATA_QUALITY

        if any(kw in msg for kw in ["cluster", "node", "memory", "disk", "oom", "connection"]):
            return FailureType.INFRASTRUCTURE

        if any(kw in msg for kw in ["network", "throttl", "rate limit", "503", "retry"]):
            return FailureType.TRANSIENT

        return FailureType.UNKNOWN

    def decide(self, event: PipelineEvent, failure_type: FailureType) -> AgentDecision:
        """Decide what action to take based on the failure classification."""
        retries = self.retry_counts.get(event.stage, 0)

        if event.status == "success":
            self.retry_counts.pop(event.stage, None)
            return AgentDecision(
                event=event,
                failure_type=failure_type,
                action=AgentAction.SKIP,
                reason="Stage succeeded — no action needed",
            )

        if failure_type == FailureType.TRANSIENT and retries < self.max_retries:
            return AgentDecision(
                event=event,
                failure_type=failure_type,
                action=AgentAction.RETRY,
                reason=f"Transient failure, retry {retries + 1}/{self.max_retries}",
                retry_count=retries + 1,
                max_retries=self.max_retries,
            )

        if failure_type == FailureType.TIMEOUT and retries < self.max_retries:
            return AgentDecision(
                event=event,
                failure_type=failure_type,
                action=AgentAction.RETRY,
                reason=f"Timeout, retry {retries + 1}/{self.max_retries} with backoff",
                retry_count=retries + 1,
                max_retries=self.max_retries,
            )

        if failure_type == FailureType.INFRASTRUCTURE and retries < 2:
            return AgentDecision(
                event=event,
                failure_type=failure_type,
                action=AgentAction.RESTART,
                reason="Infrastructure failure — restarting with fresh cluster",
                retry_count=retries + 1,
            )

        if failure_type == FailureType.DATA_QUALITY:
            return AgentDecision(
                event=event,
                failure_type=failure_type,
                action=AgentAction.QUARANTINE,
                reason="Data quality failure — quarantining bad batch",
            )

        if failure_type == FailureType.SCHEMA_DRIFT:
            return AgentDecision(
                event=event,
                failure_type=failure_type,
                action=AgentAction.ALERT,
                reason="Schema drift detected — alerting for human review",
            )

        return AgentDecision(
            event=event,
            failure_type=failure_type,
            action=AgentAction.ESCALATE,
            reason=f"Max retries ({self.max_retries}) exceeded or unknown failure",
        )

    def act(self, decision: AgentDecision) -> dict:
        """Execute the decided action and return the result."""
        self.decision_log.append(decision)
        stage = decision.event.stage

        logger.info(
            "[ACT] %s → %s (%s)", stage, decision.action.value, decision.reason
        )

        if decision.action == AgentAction.RETRY:
            self.retry_counts[stage] = decision.retry_count
            wait_time = self.backoff_base ** decision.retry_count
            logger.info("[ACT] Waiting %.1fs before retry", wait_time)
            return {
                "action": "retry",
                "stage": stage,
                "wait_seconds": wait_time,
                "retry_number": decision.retry_count,
            }

        if decision.action == AgentAction.RESTART:
            self.retry_counts[stage] = decision.retry_count
            return {"action": "restart", "stage": stage, "fresh_cluster": True}

        if decision.action == AgentAction.QUARANTINE:
            batch_id = decision.event.metadata.get("batch_id", "unknown")
            return {
                "action": "quarantine",
                "stage": stage,
                "batch_id": batch_id,
                "target": f"staging.quarantine_{stage}",
            }

        if decision.action in (AgentAction.ALERT, AgentAction.ESCALATE):
            message = (
                f"Pipeline {self.pipeline_name} — {decision.action.value}\n"
                f"Stage: {stage}\n"
                f"Failure: {decision.failure_type.value}\n"
                f"Reason: {decision.reason}\n"
                f"Error: {decision.event.error_message}"
            )
            if self.alert_callback:
                self.alert_callback(stage, message)
            return {"action": decision.action.value, "stage": stage, "message": message}

        return {"action": "no_op", "stage": stage}

    def handle_event(self, event: PipelineEvent) -> dict:
        """Full OODA loop: observe → orient → decide → act."""
        self.observe(event)
        failure_type = self.orient(event)
        decision = self.decide(event, failure_type)
        return self.act(decision)

    def get_summary(self) -> dict:
        """Return a summary of the agent's activity."""
        return {
            "pipeline": self.pipeline_name,
            "total_events": len(self.event_history),
            "total_decisions": len(self.decision_log),
            "active_retries": dict(self.retry_counts),
            "actions_taken": [
                {
                    "stage": d.event.stage,
                    "action": d.action.value,
                    "failure_type": d.failure_type.value,
                    "reason": d.reason,
                }
                for d in self.decision_log[-10:]
            ],
        }
