"""Integration tests: the agent loop operating the real PySpark pipeline.

Each test is an incident replay — inject a fault, let the agent run through
the harness, and assert on both the outcome and the data left on disk.
"""

import json
from pathlib import Path

import pytest

from agents.harness import approve_all
from agents.harness.pipeline_toolkit import RESCUED_COLUMN, FaultInjector
from agents.loop import LoopStatus, MonitoringLoop
from agents.loop.operator import make_remediation, run_agentic_pipeline
from agents.sla.sla_agent import SLADefinition, SLAMonitorAgent


def no_sleep(_seconds):
    pass


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "agentic_output"


def run(spark, output_dir, suppliers, nations, **kwargs):
    kwargs.setdefault("approver", approve_all)
    return run_agentic_pipeline(spark, output_dir, suppliers, nations, sleep=no_sleep, **kwargs)


class TestAgenticPipeline:
    def test_happy_path_publishes(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        result, toolkit = run(spark, output_dir, sample_supplier_data, sample_nation_data)

        assert result.status == LoopStatus.SUCCEEDED
        assert toolkit.state.silver.count() == 10
        for path in toolkit.state.output_paths.values():
            assert Path(path).exists()

    def test_transient_failure_recovers(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        faults = FaultInjector({"run_silver": [RuntimeError("503 Service Unavailable: throttled")]})
        result, _ = run(spark, output_dir, sample_supplier_data, sample_nation_data, faults=faults)

        assert result.status == LoopStatus.SUCCEEDED
        report = result.incident_report()
        assert report["failed_tool"] == "run_silver"
        assert report["recovery_actions"][0] == "run_silver"

    def test_empty_batch_is_quarantined(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        alerts = []
        empty = {k: [] for k in sample_supplier_data}
        result, toolkit = run(spark, output_dir, empty, sample_nation_data, alert_sink=alerts.append)

        assert result.status == LoopStatus.ESCALATED
        assert not toolkit.state.output_paths
        record = toolkit.state.quarantined[0]
        reason = json.loads((Path(record["path"]) / "reason.json").read_text())
        assert reason["stage"] == "bronze"
        assert "row_count_range" in reason["reason"]
        assert alerts and "quarantined" in alerts[0]

    def test_additive_column_rescued(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        data = {**sample_supplier_data, "s_comment": ["new upstream field"] * 10}
        result, toolkit = run(spark, output_dir, data, sample_nation_data)

        assert result.status == LoopStatus.SUCCEEDED
        rescued = toolkit.state.suppliers_bronze.select(RESCUED_COLUMN).first()[0]
        assert json.loads(rescued) == {"s_comment": "new upstream field"}
        assert RESCUED_COLUMN not in toolkit.state.silver.columns

    def test_dropped_column_escalates_before_ingest(
        self, spark, sample_supplier_data, sample_nation_data, output_dir
    ):
        data = {k: v for k, v in sample_supplier_data.items() if k != "s_phone"}
        result, toolkit = run(spark, output_dir, data, sample_nation_data)

        assert result.status == LoopStatus.ESCALATED
        assert [s.action.tool for s in result.steps] == ["inspect_source_schema", "escalate"]
        assert toolkit.state.suppliers_bronze is None

    def test_publish_requires_approval(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        result, toolkit = run(
            spark, output_dir, sample_supplier_data, sample_nation_data,
            approver=lambda tool, args: False,
        )

        assert result.status == LoopStatus.BLOCKED
        assert not (output_dir / "gold").exists()
        assert "run_gold" in toolkit.state.completed

    def test_audit_trail_written(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        audit_path = output_dir / "audit" / "actions.jsonl"
        result, _ = run(spark, output_dir, sample_supplier_data, sample_nation_data, audit_path=audit_path)

        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert len(records) == len(result.steps)
        publish = next(r for r in records if r["tool_name"] == "publish_outputs")
        assert publish["permission"] == "destructive" and publish["status"] == "ok"


class TestMonitoringLoopIntegration:
    def test_stale_sla_triggers_agentic_remediation(
        self, spark, sample_supplier_data, sample_nation_data, output_dir
    ):
        sla = SLAMonitorAgent(SLADefinition("supplier_gold", max_freshness_minutes=60))
        remediate = make_remediation(
            spark, output_dir,
            load_batch=lambda: (sample_supplier_data, sample_nation_data),
            approver=approve_all, sleep=no_sleep,
        )
        monitor = MonitoringLoop(sla, remediate, sleep=no_sleep)

        first, second = monitor.run(cycles=2)

        assert first.remediated and first.run.success
        assert first.run.output_rows == 10
        assert second.sla_action == "pass" and not second.remediated
