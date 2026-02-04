"""Pipeline metrics tracking."""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """A single metric record."""
    timestamp: str
    pipeline_name: str
    run_id: str
    stage: str
    dataset_name: str
    metric: str
    value: Any


class PipelineTracker:
    """Track metrics and lineage throughout the pipeline."""

    def __init__(self, pipeline_name: str, log_file: Optional[Path] = None):
        """
        Initialize the tracker.

        Args:
            pipeline_name: Name of the pipeline
            log_file: Path to save metrics CSV
        """
        self.pipeline_name = pipeline_name
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = log_file
        self.metrics: List[MetricRecord] = []
        self.start_time = datetime.now()

    def track(
        self,
        stage: str,
        dataset_name: str,
        metric: str,
        value: Any
    ) -> None:
        """
        Track a metric.

        Args:
            stage: Pipeline stage (extract, transform, validate, load)
            dataset_name: Name of the dataset
            metric: Metric name (row_count, column_count, etc.)
            value: Metric value
        """
        record = MetricRecord(
            timestamp=datetime.now().isoformat(),
            pipeline_name=self.pipeline_name,
            run_id=self.run_id,
            stage=stage,
            dataset_name=dataset_name,
            metric=metric,
            value=value
        )
        self.metrics.append(record)
        logger.info(f"[{stage.upper()}] {dataset_name}.{metric} = {value}")

    def track_dataframe(self, stage: str, name: str, df: pd.DataFrame) -> None:
        """
        Track DataFrame metrics.

        Args:
            stage: Pipeline stage
            name: Dataset name
            df: Pandas DataFrame
        """
        self.track(stage, name, "row_count", len(df))
        self.track(stage, name, "column_count", len(df.columns))

    def track_dataframes(self, stage: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Track metrics for multiple DataFrames."""
        for name, df in dataframes.items():
            self.track_dataframe(stage, name, df)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked metrics."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "duration_seconds": duration,
            "total_metrics": len(self.metrics),
            "stages": list(set(m.stage for m in self.metrics))
        }

    def save(self) -> None:
        """Save metrics to CSV file."""
        if not self.metrics or not self.log_file:
            return

        # Convert to DataFrame
        records = [
            {
                "timestamp": m.timestamp,
                "pipeline_name": m.pipeline_name,
                "run_id": m.run_id,
                "stage": m.stage,
                "dataset_name": m.dataset_name,
                "metric": m.metric,
                "value": str(m.value)
            }
            for m in self.metrics
        ]
        metrics_df = pd.DataFrame(records)

        # Ensure parent directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Append or create
        if self.log_file.exists():
            existing_df = pd.read_csv(self.log_file)
            metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)

        metrics_df.to_csv(self.log_file, index=False)
        logger.info(f"Saved {len(self.metrics)} metrics to {self.log_file}")
