"""Pipeline toolkit — binds the Medallion pipeline to harness tools.

This is where the abstract harness meets the real data platform. Each tool
wraps one step of `MedallionPipeline` (or one of the existing agents) and
keeps the intermediate DataFrames in a per-run state object, so an agent can
drive the pipeline stage by stage instead of calling one opaque `run()`.

Tool tiers:
    READ         inspect_source_schema, profile_silver_quality,
                 get_pipeline_state, escalate (alerts, never mutates data)
    WRITE        run_bronze, run_bronze_gate, run_silver, run_silver_gate,
                 run_gold, quarantine_batch
    DESTRUCTIVE  publish_outputs (overwrites Silver/Gold paths; gated)

Stage tools enforce their own preconditions (Silver needs Bronze, publish
needs Gold), so a confused planner gets a clear error instead of a bad write.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, StructField, StructType

from agents.harness.tools import Permission, Tool, ToolRegistry
from agents.quality.quality_agent import DataQualityAgent
from agents.schema.schema_agent import DriftAction, DriftType, SchemaDrift, SchemaDriftAgent
from pipeline.bronze.ingestion import create_bronze_schemas
from pipeline.orchestrator.pipeline_runner import MedallionPipeline

logger = logging.getLogger(__name__)

RESCUED_COLUMN = "_rescued_data"

# Ordered Medallion stages, used by policies to plan the happy path.
STAGE_TOOLS = (
    "run_bronze",
    "run_bronze_gate",
    "run_silver",
    "run_silver_gate",
    "run_gold",
    "publish_outputs",
)


class PreconditionError(RuntimeError):
    """A stage was invoked before the stage it depends on succeeded."""


@dataclass
class RunState:
    """Intermediate artefacts of one agent-driven pipeline run."""

    suppliers_bronze: Optional[DataFrame] = None
    nations_bronze: Optional[DataFrame] = None
    silver: Optional[DataFrame] = None
    gold: dict[str, DataFrame] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)
    quarantined: list[dict] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    rescued_columns: list[str] = field(default_factory=list)
    quality_summary: dict = field(default_factory=dict)


class FaultInjector:
    """Raises scripted exceptions before a tool's handler runs.

    Used by tests and the incident-replay demo to prove the loop recovers
    from transient errors, bad batches, and schema drift.

        FaultInjector({"run_silver": [ConnectionError("503 throttled")]})
    """

    def __init__(self, faults: Optional[dict[str, list[Exception]]] = None):
        self._faults = {k: list(v) for k, v in (faults or {}).items()}

    def maybe_raise(self, tool_name: str) -> None:
        queue = self._faults.get(tool_name)
        if queue:
            raise queue.pop(0)


class PipelineToolkit:
    """Exposes a MedallionPipeline run as a bounded set of agent tools.

    Args:
        pipeline: The pipeline whose stages the tools drive.
        supplier_data: Raw supplier batch (column-oriented dict).
        nation_data: Raw nation batch (column-oriented dict).
        quality_agent: Shared anomaly detector; pass the same instance across
            runs so its statistical baseline accumulates.
        faults: Optional fault injector for incident replay.
        alert_sink: Receives escalation messages (pager, Slack, ticket).
    """

    def __init__(
        self,
        pipeline: MedallionPipeline,
        supplier_data: dict[str, list],
        nation_data: dict[str, list],
        quality_agent: Optional[DataQualityAgent] = None,
        faults: Optional[FaultInjector] = None,
        alert_sink: Optional[Callable[[str], None]] = None,
    ):
        self.pipeline = pipeline
        self.supplier_data = supplier_data
        self.nation_data = nation_data
        self.quality_agent = quality_agent or DataQualityAgent(
            dataset_name="silver_suppliers",
            key_column="supplier_id",
            critical_columns=["supplier_name", "nation_name"],
        )
        self.faults = faults or FaultInjector()
        self.alert_sink = alert_sink
        self.state = RunState()
        self._schemas = create_bronze_schemas()

    # ── Registry ──────────────────────────────────────────────────────

    def registry(self) -> ToolRegistry:
        stage_arg = {
            "type": "object",
            "properties": {
                "stage": {"type": "string", "enum": ["bronze", "silver"]},
                "reason": {"type": "string"},
            },
            "required": ["stage", "reason"],
        }
        return ToolRegistry([
            Tool("inspect_source_schema",
                 "Compare the incoming supplier batch's columns to the registered Bronze "
                 "schema. Returns the schema agent's verdict: accept, evolve (additive "
                 "columns are rescued into _rescued_data) or reject (breaking change).",
                 self._wrap("inspect_source_schema", self.inspect_source_schema)),
            Tool("run_bronze",
                 "Ingest the raw supplier and nation batches into the Bronze layer with "
                 "audit columns. Must run first.",
                 self._wrap("run_bronze", self.run_bronze), Permission.WRITE),
            Tool("run_bronze_gate",
                 "Run the Bronze data quality gate. Fails if the batch is empty or audit "
                 "columns are missing.",
                 self._wrap("run_bronze_gate", self.run_bronze_gate), Permission.WRITE),
            Tool("run_silver",
                 "Deduplicate, cleanse, join and conform Bronze data into Silver.",
                 self._wrap("run_silver", self.run_silver), Permission.WRITE),
            Tool("run_silver_gate",
                 "Run the Silver data quality gate (not-null, uniqueness, row count, "
                 "nation cardinality).",
                 self._wrap("run_silver_gate", self.run_silver_gate), Permission.WRITE),
            Tool("profile_silver_quality",
                 "Statistical anomaly check of Silver against the learned baseline "
                 "(z-scores on row count, null rates, key distinctness).",
                 self._wrap("profile_silver_quality", self.profile_silver_quality)),
            Tool("run_gold",
                 "Build the Gold aggregate tables from Silver.",
                 self._wrap("run_gold", self.run_gold), Permission.WRITE),
            Tool("publish_outputs",
                 "Overwrite the published Silver and Gold Parquet outputs. Requires "
                 "approval.",
                 self._wrap("publish_outputs", self.publish_outputs),
                 Permission.DESTRUCTIVE, idempotent=False),
            Tool("quarantine_batch",
                 "Write the offending batch for a stage to the quarantine area with a "
                 "reason, so it can be inspected and replayed later.",
                 self._wrap("quarantine_batch", self.quarantine_batch),
                 Permission.WRITE, input_schema=stage_arg),
            Tool("get_pipeline_state",
                 "Completed stages, row counts, quarantined batches and output paths.",
                 self._wrap("get_pipeline_state", self.get_pipeline_state)),
            Tool("escalate",
                 "Hand the incident to a human with a root-cause summary. Ends the run.",
                 self._wrap("escalate", self.escalate),
                 input_schema={
                     "type": "object",
                     "properties": {"reason": {"type": "string"}},
                     "required": ["reason"],
                 }),
        ])

    def _wrap(self, name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
        def handler(**kwargs) -> dict:
            self.faults.maybe_raise(name)
            return fn(**kwargs)
        return handler

    # ── READ tools ────────────────────────────────────────────────────

    def inspect_source_schema(self) -> dict:
        expected = self._schemas["supplier"]
        agent = SchemaDriftAgent("tpch_supplier", expected)
        expected_names = {f.name for f in expected.fields}
        incoming = list(self.supplier_data)

        drifts = [
            SchemaDrift(DriftType.ADDITIVE, name, action=DriftAction.EVOLVE)
            for name in incoming if name not in expected_names
        ] + [
            SchemaDrift(DriftType.COLUMN_DROPPED, f.name,
                        expected_type=str(f.dataType), action=DriftAction.REJECT)
            for f in expected.fields if f.name not in incoming
        ]
        agent.drift_history.extend(drifts)
        verdict = agent.decide(drifts)

        if verdict["action"] == "evolve":
            self.state.rescued_columns = [d.column_name for d in drifts]
        return verdict

    def profile_silver_quality(self) -> dict:
        silver = self._require(self.state.silver, "run_silver", "profile_silver_quality")
        metrics = self.quality_agent.analyze(silver)
        decision = self.quality_agent.decide(metrics)
        decision["metrics"] = {m.metric_name: m.level.value for m in metrics}
        return decision

    def get_pipeline_state(self) -> dict:
        s = self.state
        return {
            "run_id": self.pipeline.metrics.run_id,
            "completed": list(s.completed),
            "silver_rows": s.silver.count() if s.silver is not None else None,
            "gold_tables": {k: v.count() for k, v in s.gold.items()},
            "quarantined": list(s.quarantined),
            "rescued_columns": list(s.rescued_columns),
            "output_paths": dict(s.output_paths),
        }

    def escalate(self, reason: str) -> dict:
        message = f"[{self.pipeline.pipeline_name}/{self.pipeline.metrics.run_id}] {reason}"
        self.state.escalations.append(reason)
        if self.alert_sink:
            self.alert_sink(message)
        logger.error("[ESCALATE] %s", message)
        return {"escalated": True, "message": message}

    # ── WRITE tools ───────────────────────────────────────────────────

    def run_bronze(self) -> dict:
        supplier_data, schemas = self._apply_rescue(self.supplier_data)
        suppliers, nations = self.pipeline.run_bronze(supplier_data, self.nation_data, schemas)
        self.state.suppliers_bronze, self.state.nations_bronze = suppliers, nations
        self._complete("run_bronze")
        return {"suppliers": suppliers.count(), "nations": nations.count(),
                "rescued_columns": list(self.state.rescued_columns)}

    def run_bronze_gate(self) -> dict:
        suppliers = self._require(self.state.suppliers_bronze, "run_bronze", "run_bronze_gate")
        checker = self.pipeline.run_bronze_quality_gate(suppliers, self.state.nations_bronze)
        self._complete("run_bronze_gate")
        return checker.summary()

    def run_silver(self) -> dict:
        self._require_done("run_bronze_gate", "run_silver")
        silver = self.pipeline.run_silver(self.state.suppliers_bronze, self.state.nations_bronze)
        self.state.silver = silver
        self._complete("run_silver")
        return {"rows": silver.count(), "columns": len(silver.columns)}

    def run_silver_gate(self) -> dict:
        silver = self._require(self.state.silver, "run_silver", "run_silver_gate")
        checker = self.pipeline.run_silver_quality_gate(silver)
        self.state.quality_summary = checker.summary()
        self._complete("run_silver_gate")
        return checker.summary()

    def run_gold(self) -> dict:
        self._require_done("run_silver_gate", "run_gold")
        self.state.gold = self.pipeline.run_gold(self.state.silver)
        self._complete("run_gold")
        return {name: df.count() for name, df in self.state.gold.items()}

    def quarantine_batch(self, stage: str, reason: str) -> dict:
        df = self.state.suppliers_bronze if stage == "bronze" else self.state.silver
        target = self.pipeline.output_dir / "quarantine" / stage / self.pipeline.metrics.run_id
        rows = 0
        if df is not None:
            rows = df.count()
            df.write.mode("overwrite").parquet(str(target / "data"))
        target.mkdir(parents=True, exist_ok=True)
        (target / "reason.json").write_text(
            json.dumps({"stage": stage, "reason": reason, "rows": rows}, indent=2)
        )
        record = {"stage": stage, "reason": reason, "rows": rows, "path": str(target)}
        self.state.quarantined.append(record)
        return record

    # ── DESTRUCTIVE tools ─────────────────────────────────────────────

    def publish_outputs(self) -> dict:
        self._require_done("run_gold", "publish_outputs")
        paths = self.pipeline.write_outputs(self.state.silver, self.state.gold)
        self.state.output_paths = paths
        self._complete("publish_outputs")
        return paths

    # ── Helpers ───────────────────────────────────────────────────────

    def _apply_rescue(
        self, data: dict[str, list]
    ) -> tuple[dict[str, list], Optional[dict[str, StructType]]]:
        """Auto Loader-style rescue: pack unknown columns into one JSON column."""
        extras = [c for c in self.state.rescued_columns if c in data]
        if not extras:
            return data, None
        known = {k: v for k, v in data.items() if k not in extras}
        n_rows = len(next(iter(data.values()), []))
        known[RESCUED_COLUMN] = [
            json.dumps({c: data[c][i] for c in extras}, default=str) for i in range(n_rows)
        ]
        schema = StructType(
            list(self._schemas["supplier"].fields)
            + [StructField(RESCUED_COLUMN, StringType(), True)]
        )
        return known, {"supplier": schema}

    def _complete(self, step: str) -> None:
        if step not in self.state.completed:
            self.state.completed.append(step)

    def _require_done(self, step: str, caller: str) -> None:
        if step not in self.state.completed:
            raise PreconditionError(f"{step} must succeed before {caller}")

    def _require(self, value, step: str, caller: str):
        if value is None:
            raise PreconditionError(f"{step} must succeed before {caller}")
        return value
