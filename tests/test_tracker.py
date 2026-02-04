"""Tests for the pipeline tracker module."""

import pytest
import pandas as pd
from pathlib import Path

from src.utils.tracker import PipelineTracker, MetricRecord


class TestPipelineTracker:
    """Tests for PipelineTracker."""

    def test_init(self):
        tracker = PipelineTracker(pipeline_name="test")
        assert tracker.pipeline_name == "test"
        assert tracker.run_id.startswith("run_")
        assert tracker.metrics == []

    def test_track(self):
        tracker = PipelineTracker(pipeline_name="test")
        tracker.track("extract", "dataset1", "row_count", 100)
        assert len(tracker.metrics) == 1
        assert tracker.metrics[0].stage == "extract"
        assert tracker.metrics[0].metric == "row_count"
        assert tracker.metrics[0].value == 100

    def test_track_dataframe(self):
        tracker = PipelineTracker(pipeline_name="test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tracker.track_dataframe("extract", "mydata", df)
        assert len(tracker.metrics) == 2
        row_metric = next(m for m in tracker.metrics if m.metric == "row_count")
        col_metric = next(m for m in tracker.metrics if m.metric == "column_count")
        assert row_metric.value == 3
        assert col_metric.value == 2

    def test_track_dataframes(self):
        tracker = PipelineTracker(pipeline_name="test")
        dfs = {
            "df1": pd.DataFrame({"a": [1]}),
            "df2": pd.DataFrame({"b": [2, 3]}),
        }
        tracker.track_dataframes("extract", dfs)
        assert len(tracker.metrics) == 4  # 2 metrics per DF

    def test_get_summary(self):
        tracker = PipelineTracker(pipeline_name="test_pipe")
        tracker.track("extract", "ds", "row_count", 10)
        tracker.track("transform", "ds", "row_count", 8)
        summary = tracker.get_summary()
        assert summary["pipeline_name"] == "test_pipe"
        assert summary["total_metrics"] == 2
        assert set(summary["stages"]) == {"extract", "transform"}
        assert summary["duration_seconds"] >= 0

    def test_save(self, tmp_path):
        log_file = tmp_path / "metrics.csv"
        tracker = PipelineTracker(pipeline_name="test", log_file=log_file)
        tracker.track("extract", "ds", "row_count", 50)
        tracker.save()

        assert log_file.exists()
        saved = pd.read_csv(log_file)
        assert len(saved) == 1
        assert saved.iloc[0]["metric"] == "row_count"

    def test_save_appends(self, tmp_path):
        log_file = tmp_path / "metrics.csv"

        tracker1 = PipelineTracker(pipeline_name="test", log_file=log_file)
        tracker1.track("extract", "ds", "row_count", 10)
        tracker1.save()

        tracker2 = PipelineTracker(pipeline_name="test", log_file=log_file)
        tracker2.track("load", "ds", "row_count", 8)
        tracker2.save()

        saved = pd.read_csv(log_file)
        assert len(saved) == 2

    def test_save_no_metrics(self, tmp_path):
        log_file = tmp_path / "metrics.csv"
        tracker = PipelineTracker(pipeline_name="test", log_file=log_file)
        tracker.save()
        assert not log_file.exists()

    def test_save_no_log_file(self):
        tracker = PipelineTracker(pipeline_name="test", log_file=None)
        tracker.track("extract", "ds", "row_count", 1)
        tracker.save()  # Should not raise

    def test_optional_log_file_type_hint(self):
        """Verify PipelineTracker accepts None for log_file without type errors."""
        tracker = PipelineTracker(pipeline_name="test")
        assert tracker.log_file is None
