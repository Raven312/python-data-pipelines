"""Unit tests for the recurring monitoring loop."""

from datetime import datetime, timedelta, timezone

import pytest

from agents.loop.monitor import MonitoringLoop, RunReport
from agents.sla.sla_agent import SLADefinition, SLAMonitorAgent


def fresh():
    return datetime.now(timezone.utc)


def stale(minutes=180):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


class Remediator:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> RunReport:
        self.calls += 1
        success, status = self.outcomes.pop(0)
        return RunReport(success, 5.0, 10 if success else 0, 10, status=status, reason="test")


@pytest.fixture
def sla():
    return SLAMonitorAgent(SLADefinition("suppliers_gold", max_freshness_minutes=60))


def test_healthy_tick_does_nothing(sla):
    remediate = Remediator([])
    report = MonitoringLoop(sla, remediate, last_success_at=fresh()).tick()
    assert report.sla_action == "pass"
    assert not report.remediated and remediate.calls == 0


def test_stale_data_triggers_remediation(sla):
    remediate = Remediator([(True, "succeeded")])
    monitor = MonitoringLoop(sla, remediate, last_success_at=stale())
    report = monitor.tick()
    assert "freshness" in report.breached_metrics
    assert report.remediated and report.run.success
    assert sla.run_history[-1]["success"] is True


def test_successful_remediation_restores_freshness(sla):
    remediate = Remediator([(True, "succeeded")])
    monitor = MonitoringLoop(sla, remediate, last_success_at=stale(), sleep=lambda s: None)
    monitor.run(cycles=2)
    assert remediate.calls == 1
    assert monitor.history[1].sla_action == "pass"


def test_never_run_pipeline_is_stale(sla):
    remediate = Remediator([(True, "succeeded")])
    assert MonitoringLoop(sla, remediate).tick().remediated


def test_repeated_failures_escalate_and_pause(sla):
    alerts = []
    remediate = Remediator([(False, "failed")] * 2)
    monitor = MonitoringLoop(sla, remediate, max_failed_remediations=2,
                             alert_sink=alerts.append, last_success_at=stale(),
                             sleep=lambda s: None)
    reports = monitor.run(cycles=4)
    assert remediate.calls == 2
    assert reports[1].escalated
    assert not reports[2].remediated and not reports[3].remediated
    assert any("pausing self-healing" in a for a in alerts)


def test_agent_escalation_pauses_immediately(sla):
    remediate = Remediator([(False, "escalated")])
    monitor = MonitoringLoop(sla, remediate, last_success_at=stale())
    assert monitor.tick().escalated
    assert not monitor.tick().remediated


def test_acknowledge_resumes_self_healing(sla):
    remediate = Remediator([(False, "escalated"), (True, "succeeded")])
    monitor = MonitoringLoop(sla, remediate, last_success_at=stale())
    monitor.tick()
    monitor.acknowledge()
    assert monitor.tick().run.success


def test_non_remediable_breach_alerts_only():
    sla = SLAMonitorAgent(SLADefinition("suppliers_gold", max_latency_seconds=10))
    sla.record_run(True, 15.0, 10, 10, fresh())
    alerts = []
    remediate = Remediator([])
    report = MonitoringLoop(sla, remediate, alert_sink=alerts.append, last_success_at=fresh()).tick()
    assert report.breached_metrics == ["latency"]
    assert remediate.calls == 0
    assert alerts and "latency" in alerts[0]


def test_run_sleeps_between_cycles(sla):
    sleeps = []
    monitor = MonitoringLoop(sla, Remediator([]), interval_seconds=60,
                             sleep=sleeps.append, last_success_at=fresh())
    monitor.run(cycles=3)
    assert sleeps == [60, 60]
