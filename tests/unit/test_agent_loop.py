"""Unit tests for the agent loop and rule-based policy, using fake stage tools.

The fakes stand in for Spark so these tests exercise control flow only:
retries with backoff, quarantine, escalation, guardrail blocks and budgets.
"""

import pytest

from agents.harness import AgentHarness, GuardrailPolicy, Permission, Tool, ToolRegistry, approve_all
from agents.harness.pipeline_toolkit import STAGE_TOOLS
from agents.loop import Action, AgentLoop, Finish, LoopStatus, RuleBasedPolicy
from agents.loop.operator import verify_published


class FakePipeline:
    """Scriptable stand-in for PipelineToolkit."""

    def __init__(self, faults=None, schema="accept", profile="pass"):
        self.faults = {k: list(v) for k, v in (faults or {}).items()}
        self.schema = schema
        self.profile = profile
        self.calls: list[str] = []

    def _tool(self, name, output=None):
        def handler(**kwargs):
            self.calls.append(name)
            queue = self.faults.get(name)
            if queue:
                raise queue.pop(0)
            return output if output is not None else {"tool": name, **kwargs}
        return handler

    def registry(self) -> ToolRegistry:
        stage_arg = {
            "type": "object",
            "properties": {"stage": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["stage", "reason"],
        }
        tools = [
            Tool("inspect_source_schema", "", self._tool(
                "inspect_source_schema", {"action": self.schema, "drifts": ["s_phone"]})),
            Tool("profile_silver_quality", "", self._tool(
                "profile_silver_quality", {"action": self.profile, "reason": "row_count z=4.1"})),
            Tool("quarantine_batch", "", self._tool("quarantine_batch"), Permission.WRITE,
                 input_schema=stage_arg),
            Tool("escalate", "", self._tool("escalate"), input_schema={
                "type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"],
            }),
        ]
        for name in STAGE_TOOLS:
            perm = Permission.DESTRUCTIVE if name == "publish_outputs" else Permission.WRITE
            tools.append(Tool(name, "", self._tool(name), perm))
        return ToolRegistry(tools)


def make_loop(fake, approver=approve_all, sleeps=None, **kwargs):
    harness = AgentHarness(fake.registry(), GuardrailPolicy.operator(approver=approver))
    sleep = (lambda s: sleeps.append(s)) if sleeps is not None else (lambda s: None)
    return AgentLoop(harness, RuleBasedPolicy(), sleep=sleep, verify=verify_published, **kwargs)


class TestHappyPath:
    def test_runs_all_stages_in_order(self):
        fake = FakePipeline()
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.SUCCEEDED
        assert fake.calls == [
            "inspect_source_schema", "run_bronze", "run_bronze_gate", "run_silver",
            "run_silver_gate", "profile_silver_quality", "run_gold", "publish_outputs",
        ]

    def test_incident_report_clean_run(self):
        report = make_loop(FakePipeline()).run("load").incident_report()
        assert report["incident_detected"] is False
        assert report["outcome"] == "succeeded"

    def test_transcript_ends_with_outcome(self):
        lines = make_loop(FakePipeline()).run("load").transcript()
        assert lines[-1].startswith("=> succeeded")


class TestFailureHandling:
    def test_transient_failure_retried_with_backoff(self):
        sleeps = []
        fake = FakePipeline(faults={"run_silver": [
            RuntimeError("503 Service Unavailable"), RuntimeError("rate limit exceeded"),
        ]})
        result = make_loop(fake, sleeps=sleeps).run("load")
        assert result.status == LoopStatus.SUCCEEDED
        assert fake.calls.count("run_silver") == 3
        assert sleeps == [2.0, 4.0]

    def test_retry_budget_resets_after_success(self):
        fake = FakePipeline(faults={
            "run_bronze": [RuntimeError("503")] * 3,
            "run_silver": [RuntimeError("503")] * 3,
        })
        assert make_loop(fake).run("load").status == LoopStatus.SUCCEEDED

    def test_retries_exhausted_escalates(self):
        fake = FakePipeline(faults={"run_gold": [RuntimeError("503 throttled")] * 4})
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.ESCALATED
        assert fake.calls.count("run_gold") == 4
        assert "publish_outputs" not in fake.calls

    def test_quality_failure_quarantines_then_escalates(self):
        fake = FakePipeline(faults={"run_bronze_gate": [RuntimeError("Bronze quality gate failed")]})
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.ESCALATED
        assert fake.calls[-2:] == ["quarantine_batch", "escalate"]
        quarantine = next(s for s in result.steps if s.action.tool == "quarantine_batch")
        assert quarantine.action.args["stage"] == "bronze"
        assert "run_silver" not in fake.calls

    def test_infrastructure_failure_restarts_stage(self):
        fake = FakePipeline(faults={"run_silver": [RuntimeError("executor lost: OOM")]})
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.SUCCEEDED
        assert fake.calls.count("run_silver") == 2

    def test_unknown_failure_escalates(self):
        fake = FakePipeline(faults={"run_gold": [RuntimeError("something odd")]})
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.ESCALATED
        assert "unknown" in result.reason

    def test_incident_report_for_recovered_run(self):
        fake = FakePipeline(faults={"run_silver": [RuntimeError("503")]})
        report = make_loop(fake).run("load").incident_report()
        assert report["detected_at_step"] == 4
        assert report["failed_tool"] == "run_silver"
        assert report["recovery_actions"][0] == "run_silver"


class TestSchemaAndQualityChecks:
    def test_breaking_schema_drift_escalates_before_any_write(self):
        fake = FakePipeline(schema="reject")
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.ESCALATED
        assert fake.calls == ["inspect_source_schema", "escalate"]

    def test_additive_drift_continues(self):
        assert make_loop(FakePipeline(schema="evolve")).run("load").succeeded

    def test_critical_anomaly_quarantines_silver(self):
        fake = FakePipeline(profile="quarantine")
        result = make_loop(fake).run("load")
        assert result.status == LoopStatus.ESCALATED
        assert "run_gold" not in fake.calls
        quarantine = next(s for s in result.steps if s.action.tool == "quarantine_batch")
        assert quarantine.action.args["stage"] == "silver"

    def test_warning_anomaly_continues(self):
        assert make_loop(FakePipeline(profile="warn")).run("load").succeeded


class TestStopConditions:
    def test_publish_denied_is_blocked(self):
        result = make_loop(FakePipeline(), approver=lambda name, args: False).run("load")
        assert result.status == LoopStatus.BLOCKED
        assert "approval denied" in result.reason

    def test_step_budget(self):
        result = make_loop(FakePipeline(), max_steps=3).run("load")
        assert result.status == LoopStatus.BUDGET_EXHAUSTED
        assert len(result.steps) == 3

    def test_time_budget(self):
        ticks = iter(range(100))
        fake = FakePipeline()
        harness = AgentHarness(fake.registry(), GuardrailPolicy.operator(approver=approve_all))
        loop = AgentLoop(harness, RuleBasedPolicy(), time_budget_seconds=3,
                         clock=lambda: next(ticks), sleep=lambda s: None)
        assert loop.run("load").status == LoopStatus.BUDGET_EXHAUSTED

    def test_consecutive_failures(self):
        class AlwaysRun:
            def decide(self, obs):
                return Action("run_gold")
        fake = FakePipeline(faults={"run_gold": [RuntimeError("x")] * 10})
        harness = AgentHarness(fake.registry())
        loop = AgentLoop(harness, AlwaysRun(), max_consecutive_failures=3)
        result = loop.run("load")
        assert result.status == LoopStatus.FAILED
        assert len(result.steps) == 3

    def test_false_success_claim_rejected(self):
        class Liar:
            def decide(self, obs):
                return Finish(LoopStatus.SUCCEEDED, "trust me")
        harness = AgentHarness(FakePipeline().registry())
        result = AgentLoop(harness, Liar(), verify=verify_published).run("load")
        assert result.status == LoopStatus.FAILED
        assert "success claim rejected" in result.reason

    def test_audit_summary_attached(self):
        result = make_loop(FakePipeline()).run("load")
        assert result.audit["total_actions"] == 8
        assert result.audit["blocked"] == 0


@pytest.mark.parametrize("tool", ["run_bronze", "run_silver", "run_gold"])
def test_precondition_errors_escalate(tool):
    fake = FakePipeline(faults={tool: [RuntimeError("PreconditionError: run_x must succeed first")]})
    assert make_loop(fake).run("load").status == LoopStatus.ESCALATED
