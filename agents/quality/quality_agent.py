"""Data Quality Agent — autonomous anomaly detection and remediation.

This agent continuously monitors data quality metrics and takes
autonomous action when anomalies are detected:
  - Statistical anomaly detection (Z-score on row counts, null rates)
  - Trend analysis (compare current batch to historical baseline)
  - Auto-quarantine of bad data batches
  - Self-healing: re-trigger ingestion from clean source

Decision framework:
  - Minor anomaly (warning threshold): Log + continue
  - Major anomaly (critical threshold): Quarantine + alert
  - Persistent anomaly (N consecutive): Escalate to human
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class AnomalyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """A single quality metric observation."""

    metric_name: str
    current_value: float
    historical_mean: float
    historical_stddev: float
    z_score: float
    level: AnomalyLevel
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class QualityProfile:
    """Historical quality profile for a dataset."""

    dataset_name: str
    expected_row_count_mean: float = 0.0
    expected_row_count_stddev: float = 0.0
    expected_null_rate_mean: float = 0.0
    expected_null_rate_stddev: float = 0.01
    expected_distinct_ratio_mean: float = 1.0
    expected_distinct_ratio_stddev: float = 0.05
    warning_z_threshold: float = 2.0
    critical_z_threshold: float = 3.0
    observations: list[dict] = field(default_factory=list)

    def update(self, row_count: float, null_rate: float, distinct_ratio: float) -> None:
        """Add an observation and recompute statistics."""
        self.observations.append({
            "row_count": row_count,
            "null_rate": null_rate,
            "distinct_ratio": distinct_ratio,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.observations) >= 3:
            counts = [o["row_count"] for o in self.observations]
            self.expected_row_count_mean = sum(counts) / len(counts)
            self.expected_row_count_stddev = max(
                _stddev(counts), 1.0
            )
            null_rates = [o["null_rate"] for o in self.observations]
            self.expected_null_rate_mean = sum(null_rates) / len(null_rates)
            self.expected_null_rate_stddev = max(_stddev(null_rates), 0.001)


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


class DataQualityAgent:
    """Autonomous agent for data quality monitoring and remediation.

    Monitors incoming data batches against a learned quality profile,
    detects statistical anomalies, and takes corrective action.
    """

    def __init__(
        self,
        dataset_name: str,
        key_column: str,
        critical_columns: list[str],
        profile: Optional[QualityProfile] = None,
    ):
        self.dataset_name = dataset_name
        self.key_column = key_column
        self.critical_columns = critical_columns
        self.profile = profile or QualityProfile(dataset_name)
        self.anomaly_history: list[QualityMetric] = []
        self.consecutive_anomalies = 0

    def analyze(self, df: DataFrame) -> list[QualityMetric]:
        """Analyze a DataFrame batch for quality anomalies.

        Returns a list of QualityMetric observations with anomaly levels.
        """
        metrics: list[QualityMetric] = []

        # Row count check
        row_count = df.count()
        rc_metric = self._check_metric(
            "row_count",
            float(row_count),
            self.profile.expected_row_count_mean,
            self.profile.expected_row_count_stddev,
        )
        metrics.append(rc_metric)

        # Null rate checks for critical columns
        for col in self.critical_columns:
            if col in df.columns:
                null_count = df.filter(F.col(col).isNull()).count()
                null_rate = null_count / max(row_count, 1)
                nr_metric = self._check_metric(
                    f"null_rate_{col}",
                    null_rate,
                    self.profile.expected_null_rate_mean,
                    self.profile.expected_null_rate_stddev,
                )
                metrics.append(nr_metric)

        # Distinct ratio check on key column
        if self.key_column in df.columns:
            distinct = df.select(self.key_column).distinct().count()
            distinct_ratio = distinct / max(row_count, 1)
            dr_metric = self._check_metric(
                f"distinct_ratio_{self.key_column}",
                distinct_ratio,
                self.profile.expected_distinct_ratio_mean,
                self.profile.expected_distinct_ratio_stddev,
            )
            metrics.append(dr_metric)

        # Update profile with this observation
        null_rates = [
            m.current_value for m in metrics if m.metric_name.startswith("null_rate")
        ]
        avg_null = sum(null_rates) / max(len(null_rates), 1)
        self.profile.update(float(row_count), avg_null, distinct_ratio if self.key_column in df.columns else 1.0)

        # Track anomaly streak
        has_critical = any(m.level == AnomalyLevel.CRITICAL for m in metrics)
        if has_critical:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0

        self.anomaly_history.extend(metrics)
        return metrics

    def decide(self, metrics: list[QualityMetric]) -> dict:
        """Decide what action to take based on quality analysis.

        Returns:
            Action dict with 'action', 'reason', and metadata.
        """
        critical_metrics = [m for m in metrics if m.level == AnomalyLevel.CRITICAL]
        warning_metrics = [m for m in metrics if m.level == AnomalyLevel.WARNING]

        if self.consecutive_anomalies >= 3:
            return {
                "action": "escalate",
                "reason": f"{self.consecutive_anomalies} consecutive critical anomalies",
                "metrics": [m.metric_name for m in critical_metrics],
            }

        if critical_metrics:
            return {
                "action": "quarantine",
                "reason": f"Critical anomaly in: {[m.metric_name for m in critical_metrics]}",
                "metrics": [m.metric_name for m in critical_metrics],
                "z_scores": {m.metric_name: m.z_score for m in critical_metrics},
            }

        if warning_metrics:
            return {
                "action": "warn",
                "reason": f"Warning anomaly in: {[m.metric_name for m in warning_metrics]}",
                "metrics": [m.metric_name for m in warning_metrics],
            }

        return {"action": "pass", "reason": "All metrics within normal range"}

    def _check_metric(
        self,
        name: str,
        current: float,
        mean: float,
        stddev: float,
    ) -> QualityMetric:
        """Compute Z-score and classify anomaly level."""
        if stddev == 0:
            z_score = 0.0
        else:
            z_score = abs(current - mean) / stddev

        if z_score >= self.profile.critical_z_threshold:
            level = AnomalyLevel.CRITICAL
        elif z_score >= self.profile.warning_z_threshold:
            level = AnomalyLevel.WARNING
        else:
            level = AnomalyLevel.NORMAL

        metric = QualityMetric(
            metric_name=name,
            current_value=current,
            historical_mean=mean,
            historical_stddev=stddev,
            z_score=round(z_score, 3),
            level=level,
        )

        if level != AnomalyLevel.NORMAL:
            logger.warning(
                "[DQ AGENT] %s: %s (value=%.3f, mean=%.3f, z=%.2f)",
                level.value.upper(), name, current, mean, z_score,
            )

        return metric

    def get_summary(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "total_observations": len(self.profile.observations),
            "consecutive_anomalies": self.consecutive_anomalies,
            "profile": {
                "row_count_mean": round(self.profile.expected_row_count_mean, 1),
                "row_count_stddev": round(self.profile.expected_row_count_stddev, 1),
                "null_rate_mean": round(self.profile.expected_null_rate_mean, 4),
            },
        }
