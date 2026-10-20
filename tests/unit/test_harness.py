"""Unit tests for the agent harness: tools, guardrails, hooks, audit."""

import json

import pytest

from agents.harness import (
    AgentHarness,
    AuditTrail,
    GuardrailPolicy,
    HookManager,
    Permission,
    Tool,
    ToolRegistry,
    approve_all,
)


def _registry(calls: list) -> ToolRegistry:
    def read_metrics():
        calls.append("read_metrics")
        return {"rows": 10}

    def run_stage(stage: str, attempts: int = 1):
        calls.append(f"run_stage:{stage}")
        return {"stage": stage, "attempts": attempts}

    def publish():
        calls.append("publish")
        return {"published": True}

    def explode():
        raise ValueError("boom")

    return ToolRegistry([
        Tool("read_metrics", "Read metrics", read_metrics),
        Tool("run_stage", "Run a stage", run_stage, Permission.WRITE, input_schema={
            "type": "object",
            "properties": {
                "stage": {"type": "string", "enum": ["bronze", "silver"]},
                "attempts": {"type": "integer"},
            },
            "required": ["stage"],
        }),
        Tool("publish", "Publish outputs", publish, Permission.DESTRUCTIVE),
        Tool("explode", "Always fails", explode),
    ])


@pytest.fixture
def calls():
    return []


class TestToolSchema:
    def test_valid_args(self, calls):
        tool = _registry(calls).get("run_stage")
        assert tool.validate({"stage": "bronze", "attempts": 2}) is None

    def test_missing_required(self, calls):
        assert "missing required" in _registry(calls).get("run_stage").validate({})

    def test_unknown_argument(self, calls):
        tool = _registry(calls).get("run_stage")
        assert "unknown argument" in tool.validate({"stage": "bronze", "x": 1})

    def test_wrong_type(self, calls):
        tool = _registry(calls).get("run_stage")
        assert "type integer" in tool.validate({"stage": "bronze", "attempts": "2"})

    def test_bool_is_not_integer(self, calls):
        tool = _registry(calls).get("run_stage")
        assert tool.validate({"stage": "bronze", "attempts": True}) is not None

    def test_enum(self, calls):
        assert "one of" in _registry(calls).get("run_stage").validate({"stage": "gold"})


class TestToolRegistry:
    def test_duplicate_rejected(self, calls):
        reg = _registry(calls)
        with pytest.raises(ValueError):
            reg.register(Tool("publish", "dup", lambda: None))

    def test_specs_filtered_by_permission(self, calls):
        names = {s["name"] for s in _registry(calls).specs(Permission.WRITE)}
        assert names == {"read_metrics", "run_stage", "explode"}

    def test_spec_shape(self, calls):
        spec = _registry(calls).get("run_stage").spec()
        assert set(spec) == {"name", "description", "input_schema"}


class TestGuardrails:
    def test_allowed_call_runs(self, calls):
        h = AgentHarness(_registry(calls))
        result = h.invoke("run_stage", {"stage": "bronze"})
        assert result.ok and result.output["stage"] == "bronze"
        assert calls == ["run_stage:bronze"]

    def test_unknown_tool_blocked(self, calls):
        result = AgentHarness(_registry(calls)).invoke("drop_table")
        assert result.blocked and "unknown tool" in result.error

    def test_permission_ceiling(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy.read_only())
        result = h.invoke("run_stage", {"stage": "bronze"})
        assert result.blocked and "ceiling is read" in result.error
        assert calls == []

    def test_destructive_denied_by_default(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy.operator())
        result = h.invoke("publish")
        assert result.blocked and "approval denied" in result.error
        assert "publish" not in calls

    def test_destructive_with_approval(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy.operator(approver=approve_all))
        assert h.invoke("publish").ok

    def test_approver_sees_tool_and_args(self, calls):
        seen = []
        policy = GuardrailPolicy.operator(approver=lambda name, args: seen.append((name, args)) or True)
        AgentHarness(_registry(calls), policy).invoke("publish")
        assert seen == [("publish", {})]

    def test_invalid_args_blocked_before_handler(self, calls):
        result = AgentHarness(_registry(calls)).invoke("run_stage", {"stage": 7})
        assert result.blocked and "invalid arguments" in result.error
        assert calls == []

    def test_per_tool_budget(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy(max_calls_per_tool=2))
        results = [h.invoke("read_metrics") for _ in range(3)]
        assert [r.ok for r in results] == [True, True, False]
        assert "call budget" in results[-1].error

    def test_total_budget(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy(max_total_calls=2))
        h.invoke("read_metrics")
        h.invoke("run_stage", {"stage": "silver"})
        assert "total call budget" in h.invoke("read_metrics").error

    def test_blocked_calls_do_not_consume_budget(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy(max_total_calls=1))
        h.invoke("run_stage", {})  # invalid → blocked
        assert h.invoke("read_metrics").ok

    def test_denylist(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy(denylist={"read_metrics"}))
        assert "denylisted" in h.invoke("read_metrics").error


class TestHarnessExecution:
    def test_handler_exception_becomes_error_result(self, calls):
        result = AgentHarness(_registry(calls)).invoke("explode")
        assert not result.ok and not result.blocked
        assert result.error == "ValueError: boom"

    def test_latency_recorded(self, calls):
        assert AgentHarness(_registry(calls)).invoke("read_metrics").latency_ms >= 0

    def test_tool_specs_respect_ceiling(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy.read_only())
        assert {s["name"] for s in h.tool_specs()} == {"read_metrics", "explode"}


class TestHooks:
    def test_pre_hook_can_block(self, calls):
        hooks = HookManager()
        hooks.pre_tool(lambda call: "maintenance window" if call.tool_name == "run_stage" else None)
        h = AgentHarness(_registry(calls), hooks=hooks)
        result = h.invoke("run_stage", {"stage": "bronze"})
        assert result.blocked and "maintenance window" in result.error
        assert calls == []

    def test_post_hook_sees_result(self, calls):
        seen = []
        hooks = HookManager()
        hooks.post_tool(lambda call, result: seen.append((call.tool_name, result.status, call.step)))
        h = AgentHarness(_registry(calls), hooks=hooks)
        h.invoke("read_metrics")
        h.invoke("explode")
        assert seen == [("read_metrics", "ok", 1), ("explode", "error", 2)]

    def test_error_hook_receives_exception(self, calls):
        seen = []
        hooks = HookManager()
        hooks.on_error(lambda call, exc: seen.append(type(exc).__name__))
        AgentHarness(_registry(calls), hooks=hooks).invoke("explode")
        assert seen == ["ValueError"]

    def test_failing_hook_is_swallowed(self, calls):
        hooks = HookManager()
        hooks.pre_tool(lambda call: 1 / 0)
        hooks.post_tool(lambda call, result: 1 / 0)
        assert AgentHarness(_registry(calls), hooks=hooks).invoke("read_metrics").ok


class TestAudit:
    def test_every_attempt_is_recorded(self, calls):
        h = AgentHarness(_registry(calls), GuardrailPolicy.operator())
        h.invoke("read_metrics")
        h.invoke("explode")
        h.invoke("publish")
        summary = h.audit.summary()
        assert (summary["ok"], summary["errors"], summary["blocked"]) == (1, 1, 1)
        assert [r.permission for r in h.audit.records] == ["read", "read", "destructive"]

    def test_records_share_session_id(self, calls):
        h = AgentHarness(_registry(calls), session_id="run_42")
        h.invoke("read_metrics")
        assert h.audit.records[0].session_id == "run_42"

    def test_jsonl_persistence(self, calls, tmp_path):
        path = tmp_path / "audit" / "actions.jsonl"
        h = AgentHarness(_registry(calls), audit=AuditTrail(path))
        h.invoke("read_metrics")
        h.invoke("run_stage", {"stage": "silver"})
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert [line["tool_name"] for line in lines] == ["read_metrics", "run_stage"]
        assert lines[1]["args"] == {"stage": "silver"}

    def test_large_outputs_are_truncated(self):
        reg = ToolRegistry([Tool("big", "big", lambda: "x" * 5000)])
        h = AgentHarness(reg)
        h.invoke("big")
        assert len(h.audit.records[0].output_summary) < 600
