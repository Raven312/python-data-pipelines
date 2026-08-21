"""Schema Drift Agent — detects upstream schema changes and adapts.

Monitors incoming data for schema differences and autonomously decides
whether to:
  - Accept additive changes (new columns) via schema evolution
  - Reject breaking changes (dropped/renamed columns, type changes)
  - Alert on ambiguous changes for human review

Schema change taxonomy:
  - ADDITIVE: New columns added → auto-evolve (safe)
  - TYPE_CHANGE: Column type changed → alert (potentially breaking)
  - COLUMN_DROPPED: Column removed → alert (breaking)
  - COLUMN_RENAMED: Column renamed → alert (breaking, needs mapping)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


class DriftType(Enum):
    ADDITIVE = "additive"
    TYPE_CHANGE = "type_change"
    COLUMN_DROPPED = "column_dropped"
    COLUMN_RENAMED = "column_renamed"
    NO_DRIFT = "no_drift"


class DriftAction(Enum):
    EVOLVE = "evolve"
    REJECT = "reject"
    ALERT = "alert"
    ACCEPT = "accept"


@dataclass
class SchemaDrift:
    """A single detected schema difference."""

    drift_type: DriftType
    column_name: str
    expected_type: Optional[str] = None
    actual_type: Optional[str] = None
    action: DriftAction = DriftAction.ALERT
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SchemaDriftAgent:
    """Detects and responds to upstream schema changes.

    Compares incoming DataFrame schemas against a registered expected
    schema and takes autonomous action based on the change type.
    """

    def __init__(self, dataset_name: str, expected_schema: StructType):
        self.dataset_name = dataset_name
        self.expected_schema = expected_schema
        self.drift_history: list[SchemaDrift] = []
        self._expected_fields = {f.name: f for f in expected_schema.fields}

    def detect(self, incoming_df: DataFrame) -> list[SchemaDrift]:
        """Compare incoming schema against expected and detect drifts."""
        drifts: list[SchemaDrift] = []
        incoming_fields = {f.name: f for f in incoming_df.schema.fields}

        # Check for new columns (additive)
        for name in incoming_fields:
            if name.startswith("_"):
                continue
            if name not in self._expected_fields:
                drifts.append(SchemaDrift(
                    drift_type=DriftType.ADDITIVE,
                    column_name=name,
                    actual_type=str(incoming_fields[name].dataType),
                    action=DriftAction.EVOLVE,
                ))

        # Check for dropped columns
        for name in self._expected_fields:
            if name not in incoming_fields:
                drifts.append(SchemaDrift(
                    drift_type=DriftType.COLUMN_DROPPED,
                    column_name=name,
                    expected_type=str(self._expected_fields[name].dataType),
                    action=DriftAction.REJECT,
                ))

        # Check for type changes
        for name in incoming_fields:
            if name in self._expected_fields:
                expected_type = str(self._expected_fields[name].dataType)
                actual_type = str(incoming_fields[name].dataType)
                if expected_type != actual_type:
                    drifts.append(SchemaDrift(
                        drift_type=DriftType.TYPE_CHANGE,
                        column_name=name,
                        expected_type=expected_type,
                        actual_type=actual_type,
                        action=DriftAction.ALERT,
                    ))

        self.drift_history.extend(drifts)

        if drifts:
            logger.warning(
                "[SCHEMA AGENT] %d drift(s) detected in %s: %s",
                len(drifts),
                self.dataset_name,
                [(d.drift_type.value, d.column_name) for d in drifts],
            )
        else:
            logger.info("[SCHEMA AGENT] No drift in %s", self.dataset_name)

        return drifts

    def decide(self, drifts: list[SchemaDrift]) -> dict:
        """Decide overall action based on detected drifts."""
        if not drifts:
            return {"action": "accept", "reason": "No schema drift detected", "drifts": []}

        has_breaking = any(
            d.drift_type in (DriftType.COLUMN_DROPPED, DriftType.TYPE_CHANGE)
            for d in drifts
        )
        all_additive = all(d.drift_type == DriftType.ADDITIVE for d in drifts)

        if all_additive:
            return {
                "action": "evolve",
                "reason": f"Only additive changes: {[d.column_name for d in drifts]}",
                "drifts": [d.column_name for d in drifts],
                "new_columns": [
                    {"name": d.column_name, "type": d.actual_type} for d in drifts
                ],
            }

        if has_breaking:
            return {
                "action": "reject",
                "reason": "Breaking schema change detected — requires human review",
                "drifts": [
                    {
                        "type": d.drift_type.value,
                        "column": d.column_name,
                        "expected": d.expected_type,
                        "actual": d.actual_type,
                    }
                    for d in drifts
                ],
            }

        return {
            "action": "alert",
            "reason": "Ambiguous schema changes detected",
            "drifts": [d.column_name for d in drifts],
        }

    def evolve_schema(self, new_columns: list[dict]) -> None:
        """Accept additive schema changes by updating the expected schema."""
        from pyspark.sql.types import StructField, StringType
        for col_info in new_columns:
            name = col_info["name"]
            if name not in self._expected_fields:
                new_field = StructField(name, StringType(), True)
                self.expected_schema = StructType(
                    list(self.expected_schema.fields) + [new_field]
                )
                self._expected_fields[name] = new_field
                logger.info("[SCHEMA AGENT] Evolved schema: added '%s'", name)

    def get_summary(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "expected_columns": len(self._expected_fields),
            "total_drifts_detected": len(self.drift_history),
            "drift_types": {
                dt.value: sum(1 for d in self.drift_history if d.drift_type == dt)
                for dt in DriftType
                if sum(1 for d in self.drift_history if d.drift_type == dt) > 0
            },
        }
