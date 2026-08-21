"""SLA Monitor Agent — pipeline freshness and latency tracking.

Monitors whether pipeline outputs meet their Service Level Agreements:
  - Data freshness: How old is the latest data? (e.g., max 1 hour stale)
  - Pipeline latency: How long does end-to-end processing take?
  - Availability: What % of scheduled runs succeed?
  - Completeness: Are all expected records present?

Actions on SLA breach:
  - WARNING: Log + metric emission
  - BREACH: Alert + trigger remediation pipeline
  - CRITICAL: Escalate to on-call engineer
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class SLAStatus(Enum):
    MET = "met"
    WARNING = "warning"
    BREACHED = "breached"
    CRITICAL = "critical"


@dataclass
class SLADefinition:
    """SLA contract for a pipeline or table."""

    name: str
    max_freshness_minutes: int = 60
    max_latency_seconds: int = 1800
    min_availability_pct: float = 99.0
    min_completeness_pct: float = 99.5
    warning_threshold_pct: float = 80.0


@dataclass
class SLACheck:
    """Result of a single SLA check."""

    sla_name: str
    metric: str
    current_value: float
    threshold: float
    status: SLAStatus
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SLAMonitorAgent:
    """Monitors pipeline SLA compliance and takes action on breaches.

    Tracks freshness, latency, availability, and completeness metrics
    against defined SLA contracts and triggers appropriate responses.
    """

    def __init__(self, sla: SLADefinition):
        self.sla = sla
        self.check_history: list[SLACheck] = []
        self.run_history: list[dict] = []

    def record_run(
        self,
        success: bool,
        duration_seconds: float,
        output_rows: int,
        expected_rows: int,
        completed_at: datetime,
    ) -> None:
        """Record a pipeline run for SLA tracking."""
        self.run_history.append({
            "success": success,
            "duration_seconds": duration_seconds,
            "output_rows": output_rows,
            "expected_rows": expected_rows,
            "completed_at": completed_at.isoformat(),
        })

    def check_all(self, last_data_timestamp: datetime) -> list[SLACheck]:
        """Run all SLA checks and return results."""
        checks: list[SLACheck] = []
        checks.append(self._check_freshness(last_data_timestamp))
        checks.append(self._check_latency())
        checks.append(self._check_availability())
        checks.append(self._check_completeness())

        self.check_history.extend(checks)

        for c in checks:
            log_fn = logger.info if c.status == SLAStatus.MET else logger.warning
            log_fn("[SLA] %s.%s → %s: %s", self.sla.name, c.metric, c.status.value, c.message)

        return checks

    def decide(self, checks: list[SLACheck]) -> dict:
        """Decide action based on SLA check results."""
        critical = [c for c in checks if c.status == SLAStatus.CRITICAL]
        breached = [c for c in checks if c.status == SLAStatus.BREACHED]
        warnings = [c for c in checks if c.status == SLAStatus.WARNING]

        if critical:
            return {
                "action": "escalate",
                "reason": f"Critical SLA breach: {[c.metric for c in critical]}",
                "checks": [c.metric for c in critical],
            }

        if breached:
            return {
                "action": "alert",
                "reason": f"SLA breached: {[c.metric for c in breached]}",
                "checks": [c.metric for c in breached],
            }

        if warnings:
            return {
                "action": "warn",
                "reason": f"SLA warning: {[c.metric for c in warnings]}",
                "checks": [c.metric for c in warnings],
            }

        return {"action": "pass", "reason": "All SLAs met"}

    def _check_freshness(self, last_data_timestamp: datetime) -> SLACheck:
        now = datetime.now(timezone.utc)
        if last_data_timestamp.tzinfo is None:
            last_data_timestamp = last_data_timestamp.replace(tzinfo=timezone.utc)
        stale_minutes = (now - last_data_timestamp).total_seconds() / 60

        threshold = self.sla.max_freshness_minutes
        warning_threshold = threshold * (self.sla.warning_threshold_pct / 100)

        if stale_minutes > threshold * 2:
            status = SLAStatus.CRITICAL
        elif stale_minutes > threshold:
            status = SLAStatus.BREACHED
        elif stale_minutes > warning_threshold:
            status = SLAStatus.WARNING
        else:
            status = SLAStatus.MET

        return SLACheck(
            sla_name=self.sla.name,
            metric="freshness",
            current_value=round(stale_minutes, 1),
            threshold=float(threshold),
            status=status,
            message=f"Data is {stale_minutes:.0f}min stale (max {threshold}min)",
        )

    def _check_latency(self) -> SLACheck:
        if not self.run_history:
            return SLACheck(
                sla_name=self.sla.name, metric="latency",
                current_value=0, threshold=float(self.sla.max_latency_seconds),
                status=SLAStatus.MET, message="No runs recorded",
            )
        last_run = self.run_history[-1]
        duration = last_run["duration_seconds"]
        threshold = self.sla.max_latency_seconds

        if duration > threshold * 2:
            status = SLAStatus.CRITICAL
        elif duration > threshold:
            status = SLAStatus.BREACHED
        elif duration > threshold * (self.sla.warning_threshold_pct / 100):
            status = SLAStatus.WARNING
        else:
            status = SLAStatus.MET

        return SLACheck(
            sla_name=self.sla.name, metric="latency",
            current_value=duration, threshold=float(threshold),
            status=status,
            message=f"Last run took {duration:.0f}s (max {threshold}s)",
        )

    def _check_availability(self) -> SLACheck:
        if not self.run_history:
            return SLACheck(
                sla_name=self.sla.name, metric="availability",
                current_value=100.0, threshold=self.sla.min_availability_pct,
                status=SLAStatus.MET, message="No runs recorded",
            )
        recent = self.run_history[-100:]
        success_count = sum(1 for r in recent if r["success"])
        availability = (success_count / len(recent)) * 100

        if availability < self.sla.min_availability_pct * 0.9:
            status = SLAStatus.CRITICAL
        elif availability < self.sla.min_availability_pct:
            status = SLAStatus.BREACHED
        elif availability < min(self.sla.min_availability_pct + 1.0, 100.0):
            status = SLAStatus.WARNING
        else:
            status = SLAStatus.MET

        return SLACheck(
            sla_name=self.sla.name, metric="availability",
            current_value=round(availability, 1),
            threshold=self.sla.min_availability_pct,
            status=status,
            message=f"Availability {availability:.1f}% (min {self.sla.min_availability_pct}%)",
        )

    def _check_completeness(self) -> SLACheck:
        if not self.run_history:
            return SLACheck(
                sla_name=self.sla.name, metric="completeness",
                current_value=100.0, threshold=self.sla.min_completeness_pct,
                status=SLAStatus.MET, message="No runs recorded",
            )
        last_run = self.run_history[-1]
        expected = last_run.get("expected_rows", 0)
        actual = last_run.get("output_rows", 0)
        completeness = (actual / max(expected, 1)) * 100

        if completeness < self.sla.min_completeness_pct * 0.9:
            status = SLAStatus.CRITICAL
        elif completeness < self.sla.min_completeness_pct:
            status = SLAStatus.BREACHED
        else:
            status = SLAStatus.MET

        return SLACheck(
            sla_name=self.sla.name, metric="completeness",
            current_value=round(completeness, 1),
            threshold=self.sla.min_completeness_pct,
            status=status,
            message=f"Completeness {completeness:.1f}% ({actual}/{expected} rows)",
        )

    def get_summary(self) -> dict:
        recent_checks = self.check_history[-20:]
        return {
            "sla_name": self.sla.name,
            "total_runs": len(self.run_history),
            "total_checks": len(self.check_history),
            "current_status": {
                c.metric: c.status.value
                for c in recent_checks[-4:]
            } if recent_checks else {},
        }
