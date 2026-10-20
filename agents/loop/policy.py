"""Rule-based policy — deterministic judgment for the agent loop.

The policy is stateless with respect to the pipeline: it decides purely from
the Observation (goal, history of actions and results). That keeps it
interchangeable with an LLM planner and makes every decision replayable.

Decision order on each step:

    1. Terminal checks      escalate succeeded → ESCALATED
                            last action blocked by guardrails → BLOCKED
    2. Pre-flight           inspect the source schema once; reject → escalate
    3. Failure handling     classify the error with PipelineOrchestrationAgent
                            (transient/timeout → retry w/ backoff,
                             infrastructure → restart, data quality →
                             quarantine, schema/unknown → escalate)
    4. Quarantine follow-up a quarantined batch always goes to a human
    5. Quality profiling    after the Silver gate, check for statistical
                            anomalies before anything reaches Gold
    6. Happy path           run the next Medallion stage not yet done
    7. Done                 every stage succeeded → SUCCEEDED
"""

from typing import Optional

from agents.harness.pipeline_toolkit import STAGE_TOOLS
from agents.loop.agent_loop import Action, Decision, Finish, LoopStatus, Observation
from agents.orchestrator.pipeline_agent import AgentAction, PipelineEvent, PipelineOrchestrationAgent

_QUARANTINE_STAGE = {
    "run_bronze": "bronze",
    "run_bronze_gate": "bronze",
    "run_silver": "silver",
    "run_silver_gate": "silver",
    "profile_silver_quality": "silver",
}


class RuleBasedPolicy:
    """Drives the Medallion stages and reacts to failures by rule.

    Args:
        stage_plan: Ordered stage tools that make up a successful run.
        max_retries: Retry budget per stage for transient failures.
        backoff_base: Exponential backoff base, in seconds.
    """

    def __init__(
        self,
        stage_plan: tuple[str, ...] = STAGE_TOOLS,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.stage_plan = stage_plan
        self.triage = PipelineOrchestrationAgent(
            "agent_loop", max_retries=max_retries, backoff_base=backoff_base
        )
        self._triaged_steps: set[int] = set()

    def decide(self, obs: Observation) -> Decision:
        last = obs.last

        # 1. Terminal checks
        if last and last.action.tool == "escalate" and last.result.ok:
            return Finish(LoopStatus.ESCALATED, last.action.args.get("reason", "escalated"))
        if last and last.result.blocked:
            return Finish(LoopStatus.BLOCKED, f"{last.action.tool}: {last.result.error}")

        # 2. Pre-flight schema inspection
        if "inspect_source_schema" in obs.tools and obs.last_result_of("inspect_source_schema") is None:
            return Action("inspect_source_schema", rationale="pre-flight schema drift check")
        schema = obs.last_result_of("inspect_source_schema")
        if schema and schema.ok and schema.output.get("action") == "reject" and not obs.last_result_of("escalate"):
            return Action(
                "escalate",
                {"reason": f"Breaking upstream schema change: {schema.output.get('drifts')}"},
                rationale="breaking schema drift needs a human",
            )

        # 3. Failure handling
        if last and not last.result.ok:
            return self._handle_failure(obs, last.index, last.action.tool, last.result.error or "")

        # 4. A quarantined batch is never silently skipped
        if last and last.action.tool == "quarantine_batch":
            return Action(
                "escalate",
                {"reason": f"Batch quarantined at {last.action.args['stage']}: {last.action.args['reason']}"},
                rationale="quarantine needs human review before replay",
            )

        # Successful stage → reset its retry counter in the triage agent
        if last and last.action.tool in self.stage_plan and last.index not in self._triaged_steps:
            self._triaged_steps.add(last.index)
            self.triage.handle_event(PipelineEvent(stage=last.action.tool, status="success"))

        # 5. Statistical quality profile between Silver and Gold
        if (
            "profile_silver_quality" in obs.tools
            and obs.succeeded("run_silver_gate")
            and obs.last_result_of("profile_silver_quality") is None
        ):
            return Action("profile_silver_quality", rationale="anomaly check before Gold")
        profile = obs.last_result_of("profile_silver_quality")
        if last and last.action.tool == "profile_silver_quality" and profile and profile.ok:
            verdict = profile.output.get("action")
            if verdict == "quarantine":
                return Action("quarantine_batch", {"stage": "silver", "reason": profile.output["reason"]},
                              rationale="critical anomaly vs. baseline")
            if verdict == "escalate":
                return Action("escalate", {"reason": profile.output["reason"]},
                              rationale="persistent anomalies")

        # 6. Happy path
        next_stage = self._next_stage(obs)
        if next_stage:
            return Action(next_stage, rationale="next Medallion stage")

        # 7. Done
        return Finish(LoopStatus.SUCCEEDED, "all stages completed")

    def _next_stage(self, obs: Observation) -> Optional[str]:
        for tool in self.stage_plan:
            if tool in obs.tools and not obs.succeeded(tool):
                return tool
        return None

    def _handle_failure(self, obs: Observation, step: int, tool: str, error: str) -> Decision:
        if tool == "escalate":
            return Finish(LoopStatus.FAILED, f"escalation itself failed: {error}")
        if tool == "quarantine_batch":
            return Action("escalate", {"reason": f"Quarantine failed: {error}"},
                          rationale="cannot isolate bad batch")

        event = PipelineEvent(stage=tool, status="failure", error_message=error)
        outcome = self.triage.handle_event(event)
        decision = self.triage.decision_log[-1]
        self._triaged_steps.add(step)

        if decision.action == AgentAction.RETRY:
            return Action(tool, rationale=decision.reason, wait_seconds=outcome["wait_seconds"])
        if decision.action == AgentAction.RESTART:
            return Action(tool, rationale=decision.reason)
        if decision.action == AgentAction.QUARANTINE and tool in _QUARANTINE_STAGE:
            return Action("quarantine_batch", {"stage": _QUARANTINE_STAGE[tool], "reason": error},
                          rationale=decision.reason)
        return Action(
            "escalate",
            {"reason": f"{tool} failed ({decision.failure_type.value}): {error}"},
            rationale=decision.reason,
        )
