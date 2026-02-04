"""Tests for the pipeline module — end-to-end."""

import pytest
import pandas as pd
from pathlib import Path

from src.pipeline import SupplierETLPipeline, PipelineResult, create_sample_data


class TestCreateSampleData:
    """Tests for the sample data helper."""

    def test_returns_two_dicts(self):
        supplier, nation = create_sample_data()
        assert isinstance(supplier, dict)
        assert isinstance(nation, dict)

    def test_supplier_keys(self):
        supplier, _ = create_sample_data()
        assert "s_suppkey" in supplier
        assert "s_name" in supplier
        assert "s_nationkey" in supplier
        assert "s_phone" in supplier

    def test_nation_keys(self):
        _, nation = create_sample_data()
        assert "n_nationkey" in nation
        assert "n_name" in nation

    def test_data_lengths_consistent(self):
        supplier, nation = create_sample_data()
        lengths = [len(v) for v in supplier.values()]
        assert len(set(lengths)) == 1
        lengths_n = [len(v) for v in nation.values()]
        assert len(set(lengths_n)) == 1


class TestSupplierETLPipeline:
    """End-to-end pipeline tests."""

    def test_pipeline_runs_successfully(self, tmp_path):
        output = tmp_path / "output.csv"
        metrics = tmp_path / "metrics.csv"
        pipeline = SupplierETLPipeline(
            output_path=output,
            metrics_path=metrics,
        )
        supplier, nation = create_sample_data()
        result = pipeline.run(supplier, nation)

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.output_rows > 0
        assert result.validation_passed is True
        assert result.output_path == output
        assert output.exists()

    def test_pipeline_output_file_content(self, tmp_path):
        output = tmp_path / "output.csv"
        pipeline = SupplierETLPipeline(output_path=output)
        supplier, nation = create_sample_data()
        pipeline.run(supplier, nation)

        df = pd.read_csv(output)
        assert "supplier_name" in df.columns
        assert "supplier_phone_number" in df.columns
        assert "supplier_nation" in df.columns
        # Names should be uppercased
        assert all(name == name.upper() for name in df["supplier_name"])

    def test_pipeline_metrics_saved(self, tmp_path):
        output = tmp_path / "output.csv"
        metrics = tmp_path / "metrics.csv"
        pipeline = SupplierETLPipeline(
            output_path=output,
            metrics_path=metrics,
        )
        supplier, nation = create_sample_data()
        pipeline.run(supplier, nation)

        assert metrics.exists()
        metrics_df = pd.read_csv(metrics)
        assert len(metrics_df) > 0

    def test_pipeline_result_fields(self, tmp_path):
        output = tmp_path / "output.csv"
        pipeline = SupplierETLPipeline(output_path=output)
        supplier, nation = create_sample_data()
        result = pipeline.run(supplier, nation)

        assert result.run_id.startswith("run_")
        assert result.duration_seconds >= 0
        assert "supplier" in result.input_rows
        assert "nation" in result.input_rows

    def test_pipeline_no_metrics_path(self, tmp_path):
        output = tmp_path / "output.csv"
        pipeline = SupplierETLPipeline(output_path=output, metrics_path=None)
        supplier, nation = create_sample_data()
        result = pipeline.run(supplier, nation)
        assert result.success is True
