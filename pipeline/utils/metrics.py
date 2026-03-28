"""Pipeline execution metrics and lineage tracking."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


@dataclass
class StageMetric:
    """Metrics captured for a single pipeline stage."""

    stage: str
    dataset: str
    row_count: int
    column_count: int
    duration_seconds: float
    timestamp: str
    metadata: dict = field(default_factory=dict)


class PipelineMetrics:
    """Collects and reports metrics across the medallion pipeline stages.

    Tracks row counts, column counts, durations, and custom metadata
    for each Bronze/Silver/Gold stage execution.
    """

    def __init__(self, pipeline_name: str, run_id: Optional[str] = None):
        self.pipeline_name = pipeline_name
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.stages: list[StageMetric] = []
        self._start_time = time.monotonic()

    @contextmanager
    def track_stage(self, stage: str, dataset: str):
        """Context manager that times a stage and yields a metadata dict.

        Usage:
            with metrics.track_stage("bronze", "suppliers") as meta:
                df = do_work()
                meta["df"] = df
                meta["extra"] = "info"
        """
        meta: dict[str, Any] = {}
        start = time.monotonic()
        yield meta
        elapsed = time.monotonic() - start

        df: Optional[DataFrame] = meta.pop("df", None)
        row_count = df.count() if df is not None else 0
        col_count = len(df.columns) if df is not None else 0

        metric = StageMetric(
            stage=stage,
            dataset=dataset,
            row_count=row_count,
            column_count=col_count,
            duration_seconds=round(elapsed, 3),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=meta,
        )
        self.stages.append(metric)
        logger.info(
            "[%s] %s — %d rows, %d cols, %.3fs",
            stage.upper(),
            dataset,
            row_count,
            col_count,
            elapsed,
        )

    @property
    def total_duration(self) -> float:
        return round(time.monotonic() - self._start_time, 3)

    def summary(self) -> dict:
        """Return a summary dict of the full pipeline run."""
        return {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "total_duration_seconds": self.total_duration,
            "stages": [
                {
                    "stage": s.stage,
                    "dataset": s.dataset,
                    "row_count": s.row_count,
                    "duration_seconds": s.duration_seconds,
                }
                for s in self.stages
            ],
        }
