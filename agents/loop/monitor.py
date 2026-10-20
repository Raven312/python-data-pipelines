"""Monitoring Loop — the recurring outer loop around the agent loop.

The agent loop handles one run. The monitoring loop runs forever on a
schedule, watches SLAs, and starts an agent run only when something needs
fixing:

    every interval:
        observe   SLA checks (freshness, latency, availability, completeness)
        decide    pass/warn → nothing · breach on a remediable metric → remediate
                  breach on any other metric → alert
        act       run the agent loop (re-run / heal the pipeline), record the
                  outcome back into the SLA agent so the next tick sees it
        guard     after N failed remediations in a row, stop retrying and
                  escalate once — no infinite self-healing loops

In production the tick is triggered by a scheduler (Databricks Workflows,
Airflow, cron); `run()` with an injected sleep is for local demos and tests.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from agents.sla.sla_agent import SLAMonitorAgent

logger = logging.getLogger(__name__)

# Breaches a fresh pipeline run can fix. Latency/availability breaches are
# symptoms a re-run would not cure, so they alert instead.
REMEDIABLE_METRICS = frozenset({"freshness", "completeness"})


@dataclass
class RunReport:
    """What a remediation run reports back to the monitor."""

    success: bool
    duration_seconds: float
    output_rows: int
    expected_rows: int
    status: str = ""
    reason: str = ""


@dataclass
class TickReport:
    cycle: int
    sla_action: str
    breached_metrics: list[str]
    remediated: bool = False
    run: Optional[RunReport] = None
    escalated: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MonitoringLoop:
    """Scheduled SLA watcher that triggers agent-driven remediation.

    Args:
        sla_agent: Evaluates SLA checks and keeps run history.
        remediate: Runs the agent loop once and returns a RunReport.
        interval_seconds: Time between ticks when using `run()`.
        max_failed_remediations: Consecutive failures before escalating.
        alert_sink: Receives alert and escalation messages.
        last_success_at: Timestamp of the last good publish (freshness input).
        sleep / now: Injected for tests.
    """

    def __init__(
        self,
        sla_agent: SLAMonitorAgent,
        remediate: Callable[[], RunReport],
        interval_seconds: float = 300,
        max_failed_remediations: int = 3,
        alert_sink: Optional[Callable[[str], None]] = None,
        last_success_at: Optional[datetime] = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.sla_agent = sla_agent
        self.remediate = remediate
        self.interval_seconds = interval_seconds
        self.max_failed_remediations = max_failed_remediations
        self.alert_sink = alert_sink
        self.last_success_at = last_success_at or datetime.fromtimestamp(0, timezone.utc)
        self.sleep = sleep
        self.now = now
        self.failed_remediations = 0
        self.escalated = False
        self.history: list[TickReport] = []

    def tick(self) -> TickReport:
        checks = self.sla_agent.check_all(self.last_success_at)
        decision = self.sla_agent.decide(checks)
        breached = [c.metric for c in checks if c.status.value in ("breached", "critical")]
        report = TickReport(len(self.history) + 1, decision["action"], breached)

        if decision["action"] in ("pass", "warn"):
            return self._record(report)

        remediable = [m for m in breached if m in REMEDIABLE_METRICS]
        if not remediable:
            self._alert(f"SLA {decision['action']}: {decision['reason']}")
            return self._record(report)

        if self.escalated:
            logger.warning("[MONITOR] remediation paused after escalation; waiting for a human")
            return self._record(report)

        run = self.remediate()
        report.remediated, report.run = True, run
        self.sla_agent.record_run(
            success=run.success,
            duration_seconds=run.duration_seconds,
            output_rows=run.output_rows,
            expected_rows=run.expected_rows,
            completed_at=self.now(),
        )

        if run.success:
            self.last_success_at = self.now()
            self.failed_remediations = 0
        else:
            self.failed_remediations += 1
            if run.status == "escalated" or self.failed_remediations >= self.max_failed_remediations:
                self.escalated = report.escalated = True
                self._alert(
                    f"Remediation failed {self.failed_remediations}x for {self.sla_agent.sla.name} "
                    f"({run.status}: {run.reason}); pausing self-healing"
                )
        return self._record(report)

    def run(self, cycles: Optional[int] = None) -> list[TickReport]:
        """Tick every `interval_seconds`; forever when cycles is None."""
        n = 0
        while cycles is None or n < cycles:
            self.tick()
            n += 1
            if cycles is None or n < cycles:
                self.sleep(self.interval_seconds)
        return self.history

    def acknowledge(self) -> None:
        """A human resolved the incident; resume self-healing."""
        self.escalated = False
        self.failed_remediations = 0

    def _record(self, report: TickReport) -> TickReport:
        self.history.append(report)
        logger.info(
            "[MONITOR] cycle=%d sla=%s breached=%s remediated=%s",
            report.cycle, report.sla_action, report.breached_metrics, report.remediated,
        )
        return report

    def _alert(self, message: str) -> None:
        logger.error("[MONITOR] %s", message)
        if self.alert_sink:
            self.alert_sink(message)
