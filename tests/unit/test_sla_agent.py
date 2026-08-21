"""Unit tests for the SLA Monitor Agent."""

import pytest
from datetime import datetime, timedelta, timezone

from agents.sla.sla_agent import SLADefinition, SLAMonitorAgent, SLAStatus


@pytest.fixture
def sla():
    return SLADefinition(
        name="silver_suppliers",
        max_freshness_minutes=60,
        max_latency_seconds=1800,
        min_availability_pct=99.0,
        min_completeness_pct=99.5,
    )


@pytest.fixture
def agent(sla):
    return SLAMonitorAgent(sla)


class TestFreshnessCheck:

    def test_fresh_data_passes(self, agent):
        recent = datetime.now(timezone.utc) - timedelta(minutes=10)
        checks = agent.check_all(recent)
        freshness = next(c for c in checks if c.metric == "freshness")
        assert freshness.status == SLAStatus.MET

    def test_stale_data_breached(self, agent):
        old = datetime.now(timezone.utc) - timedelta(minutes=90)
        checks = agent.check_all(old)
        freshness = next(c for c in checks if c.metric == "freshness")
        assert freshness.status == SLAStatus.BREACHED

    def test_very_stale_critical(self, agent):
        very_old = datetime.now(timezone.utc) - timedelta(minutes=150)
        checks = agent.check_all(very_old)
        freshness = next(c for c in checks if c.metric == "freshness")
        assert freshness.status == SLAStatus.CRITICAL


class TestLatencyCheck:

    def test_fast_run_passes(self, agent):
        agent.record_run(True, 300, 100, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        latency = next(c for c in checks if c.metric == "latency")
        assert latency.status == SLAStatus.MET

    def test_slow_run_breached(self, agent):
        agent.record_run(True, 2000, 100, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        latency = next(c for c in checks if c.metric == "latency")
        assert latency.status == SLAStatus.BREACHED


class TestAvailabilityCheck:

    def test_all_success_passes(self, agent):
        for _ in range(10):
            agent.record_run(True, 100, 100, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        avail = next(c for c in checks if c.metric == "availability")
        assert avail.status == SLAStatus.MET

    def test_failures_breach(self, agent):
        for _ in range(95):
            agent.record_run(True, 100, 100, 100, datetime.now(timezone.utc))
        for _ in range(5):
            agent.record_run(False, 100, 0, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        avail = next(c for c in checks if c.metric == "availability")
        assert avail.status in (SLAStatus.WARNING, SLAStatus.BREACHED)


class TestCompletenessCheck:

    def test_complete_passes(self, agent):
        agent.record_run(True, 100, 100, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        comp = next(c for c in checks if c.metric == "completeness")
        assert comp.status == SLAStatus.MET

    def test_incomplete_breaches(self, agent):
        agent.record_run(True, 100, 50, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        comp = next(c for c in checks if c.metric == "completeness")
        assert comp.status in (SLAStatus.BREACHED, SLAStatus.CRITICAL)


class TestDecisions:

    def test_all_met_passes(self, agent):
        agent.record_run(True, 100, 100, 100, datetime.now(timezone.utc))
        checks = agent.check_all(datetime.now(timezone.utc))
        decision = agent.decide(checks)
        assert decision["action"] == "pass"

    def test_breach_alerts(self, agent):
        agent.record_run(True, 2000, 100, 100, datetime.now(timezone.utc))
        old = datetime.now(timezone.utc) - timedelta(minutes=90)
        checks = agent.check_all(old)
        decision = agent.decide(checks)
        assert decision["action"] in ("alert", "escalate")

    def test_summary(self, agent):
        agent.record_run(True, 100, 100, 100, datetime.now(timezone.utc))
        agent.check_all(datetime.now(timezone.utc))
        summary = agent.get_summary()
        assert summary["sla_name"] == "silver_suppliers"
        assert summary["total_runs"] == 1
