"""Pipeline operator — wires toolkit, harness, policy and loop together.

    toolkit  = PipelineToolkit(pipeline, data)        # what can be done
    harness  = AgentHarness(toolkit.registry(), ...)  # under which rules
    policy   = RuleBasedPolicy() | ClaudePolicy(...)  # who decides
    loop     = AgentLoop(harness, policy, ...)        # until when

Use `run_agentic_pipeline` for a one-shot run, or `make_remediation` to plug
the operator into a MonitoringLoop.
"""

import time
from pathlib import Path
from typing import Callable, Optional, Union

from pyspark.sql import SparkSession

from agents.harness.audit import AuditTrail
from agents.harness.guardrails import Approver, GuardrailPolicy, deny_all
from agents.harness.harness import AgentHarness
from agents.harness.hooks import HookManager
from agents.harness.pipeline_toolkit import FaultInjector, PipelineToolkit
from agents.loop.agent_loop import AgentLoop, LoopResult, LoopStatus, Policy, Step
from agents.loop.monitor import RunReport
from agents.loop.policy import RuleBasedPolicy
from agents.quality.quality_agent import DataQualityAgent
from pipeline.orchestrator.pipeline_runner import MedallionPipeline

GOAL = "Load the latest supplier batch through Bronze, Silver and Gold and publish it safely."


def verify_published(steps: list[Step]) -> Optional[str]:
    """A run only counts as successful if publish_outputs actually succeeded."""
    if any(s.action.tool == "publish_outputs" and s.result.ok for s in steps):
        return None
    return "publish_outputs never succeeded"


def build_operator(
    spark: SparkSession,
    output_dir: Path,
    supplier_data: dict[str, list],
    nation_data: dict[str, list],
    *,
    policy: Union[str, Policy] = "rules",
    approver: Approver = deny_all,
    faults: Optional[FaultInjector] = None,
    quality_agent: Optional[DataQualityAgent] = None,
    hooks: Optional[HookManager] = None,
    audit_path: Optional[Path] = None,
    alert_sink: Optional[Callable[[str], None]] = None,
    max_steps: int = 25,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[AgentLoop, PipelineToolkit]:
    pipeline = MedallionPipeline(spark, output_dir)
    toolkit = PipelineToolkit(
        pipeline, supplier_data, nation_data,
        quality_agent=quality_agent, faults=faults, alert_sink=alert_sink,
    )
    harness = AgentHarness(
        toolkit.registry(),
        policy=GuardrailPolicy.operator(approver=approver),
        hooks=hooks,
        audit=AuditTrail(audit_path),
        session_id=pipeline.metrics.run_id,
    )

    if policy == "rules":
        policy = RuleBasedPolicy()
    elif policy == "claude":
        from agents.loop.claude_policy import ClaudePolicy

        policy = ClaudePolicy(harness.tool_specs())

    loop = AgentLoop(harness, policy, max_steps=max_steps, verify=verify_published, sleep=sleep)
    return loop, toolkit


def run_agentic_pipeline(*args, **kwargs) -> tuple[LoopResult, PipelineToolkit]:
    loop, toolkit = build_operator(*args, **kwargs)
    return loop.run(GOAL), toolkit


def make_remediation(
    spark: SparkSession,
    output_dir: Path,
    load_batch: Callable[[], tuple[dict, dict]],
    **operator_kwargs,
) -> Callable[[], RunReport]:
    """Adapt the operator to the MonitoringLoop's `remediate` callback.

    `load_batch` fetches the latest raw batch on each call, so every
    remediation works on fresh data rather than a stale snapshot.
    """
    def remediate() -> RunReport:
        supplier_data, nation_data = load_batch()
        result, toolkit = run_agentic_pipeline(
            spark, output_dir, supplier_data, nation_data, **operator_kwargs
        )
        expected = len(next(iter(supplier_data.values()), []))
        silver = toolkit.state.silver
        return RunReport(
            success=result.status == LoopStatus.SUCCEEDED,
            duration_seconds=result.duration_seconds,
            output_rows=silver.count() if silver is not None and result.succeeded else 0,
            expected_rows=expected,
            status=result.status.value,
            reason=result.reason,
        )

    return remediate
