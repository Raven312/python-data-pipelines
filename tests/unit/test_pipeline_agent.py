"""Unit tests for the Pipeline Orchestration Agent."""

import pytest

from agents.orchestrator.pipeline_agent import (
    AgentAction,
    FailureType,
    PipelineEvent,
    PipelineOrchestrationAgent,
)


@pytest.fixture
def agent():
    return PipelineOrchestrationAgent("test_pipeline", max_retries=3)


class TestFailureClassification:

    def test_timeout_classified(self, agent):
        event = PipelineEvent(stage="bronze", status="failure", error_message="Job timed out after 30m")
        assert agent.orient(event) == FailureType.TIMEOUT

    def test_schema_drift_classified(self, agent):
        event = PipelineEvent(stage="bronze", status="failure", error_message="Column type mismatch: expected IntegerType")
        assert agent.orient(event) == FailureType.SCHEMA_DRIFT

    def test_data_quality_classified(self, agent):
        event = PipelineEvent(stage="silver", status="failure", error_message="Validation failed: null supplier_id")
        assert agent.orient(event) == FailureType.DATA_QUALITY

    def test_infrastructure_classified(self, agent):
        event = PipelineEvent(stage="bronze", status="failure", error_message="Cluster OOM killed")
        assert agent.orient(event) == FailureType.INFRASTRUCTURE

    def test_transient_classified(self, agent):
        event = PipelineEvent(stage="bronze", status="failure", error_message="Network error: 503 retry later")
        assert agent.orient(event) == FailureType.TRANSIENT

    def test_unknown_classified(self, agent):
        event = PipelineEvent(stage="gold", status="failure", error_message="Something weird happened")
        assert agent.orient(event) == FailureType.UNKNOWN


class TestDecisionMaking:

    def test_success_skipped(self, agent):
        event = PipelineEvent(stage="bronze", status="success")
        decision = agent.decide(event, FailureType.TRANSIENT)
        assert decision.action == AgentAction.SKIP

    def test_transient_retried(self, agent):
        event = PipelineEvent(stage="bronze", status="failure")
        decision = agent.decide(event, FailureType.TRANSIENT)
        assert decision.action == AgentAction.RETRY
        assert decision.retry_count == 1

    def test_data_quality_quarantined(self, agent):
        event = PipelineEvent(stage="silver", status="failure")
        decision = agent.decide(event, FailureType.DATA_QUALITY)
        assert decision.action == AgentAction.QUARANTINE

    def test_schema_drift_alerted(self, agent):
        event = PipelineEvent(stage="bronze", status="failure")
        decision = agent.decide(event, FailureType.SCHEMA_DRIFT)
        assert decision.action == AgentAction.ALERT

    def test_infrastructure_restarted(self, agent):
        event = PipelineEvent(stage="bronze", status="failure")
        decision = agent.decide(event, FailureType.INFRASTRUCTURE)
        assert decision.action == AgentAction.RESTART

    def test_max_retries_escalates(self, agent):
        event = PipelineEvent(stage="bronze", status="failure")
        agent.retry_counts["bronze"] = 3
        decision = agent.decide(event, FailureType.TRANSIENT)
        assert decision.action == AgentAction.ESCALATE


class TestOODALoop:

    def test_handle_event_full_loop(self, agent):
        event = PipelineEvent(stage="bronze", status="failure", error_message="Network timeout")
        result = agent.handle_event(event)
        assert result["action"] == "retry"
        assert result["wait_seconds"] > 0

    def test_multiple_retries_backoff(self, agent):
        for i in range(3):
            event = PipelineEvent(stage="bronze", status="failure", error_message="503 retry")
            result = agent.handle_event(event)
            if i < 2:
                assert result["action"] == "retry"
                assert result["wait_seconds"] == agent.backoff_base ** (i + 1)

    def test_escalation_after_max_retries(self, agent):
        for i in range(4):
            event = PipelineEvent(stage="bronze", status="failure", error_message="503 retry")
            result = agent.handle_event(event)
        assert result["action"] == "escalate"

    def test_success_resets_retries(self, agent):
        agent.retry_counts["bronze"] = 2
        event = PipelineEvent(stage="bronze", status="success")
        agent.handle_event(event)
        assert "bronze" not in agent.retry_counts

    def test_alert_callback_invoked(self):
        alerts = []
        agent = PipelineOrchestrationAgent(
            "test", alert_callback=lambda s, m: alerts.append((s, m))
        )
        event = PipelineEvent(stage="bronze", status="failure", error_message="Schema mismatch: StructType")
        agent.handle_event(event)
        assert len(alerts) == 1

    def test_summary(self, agent):
        event = PipelineEvent(stage="bronze", status="failure", error_message="timeout")
        agent.handle_event(event)
        summary = agent.get_summary()
        assert summary["total_events"] == 1
        assert summary["total_decisions"] == 1
