"""Integration tests: end-to-end Medallion pipeline execution."""

import pytest
from pathlib import Path

from pipeline.orchestrator.pipeline_runner import MedallionPipeline


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "pipeline_output"


class TestMedallionPipelineE2E:
    """End-to-end integration tests for the full pipeline."""

    def test_successful_run(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert result.success is True
        assert result.error is None

    def test_input_row_counts(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert result.input_rows["suppliers"] == 10
        assert result.input_rows["nations"] == 9

    def test_silver_row_count(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert result.silver_rows == 10

    def test_gold_tables_produced(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert "supplier_by_nation" in result.gold_tables
        assert "supplier_summary" in result.gold_tables
        assert result.gold_tables["supplier_summary"] == 1

    def test_output_files_written(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert "silver_suppliers" in result.output_paths
        assert "gold_supplier_by_nation" in result.output_paths
        assert "gold_supplier_summary" in result.output_paths

        # Verify files exist on disk
        for path in result.output_paths.values():
            assert Path(path).exists()

    def test_silver_output_readable(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        silver_path = result.output_paths["silver_suppliers"]
        df = spark.read.parquet(silver_path)
        assert df.count() == 10
        assert "supplier_id" in df.columns
        assert "nation_name" in df.columns

    def test_gold_output_readable(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        gold_path = result.output_paths["gold_supplier_by_nation"]
        df = spark.read.parquet(gold_path)
        assert df.count() >= 1
        assert "supplier_count" in df.columns

    def test_quality_summary_included(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert "checkpoint" in result.quality_summary
        assert result.quality_summary["pass_rate"] == 1.0

    def test_run_id_generated(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert result.run_id.startswith("run_")

    def test_duration_measured(self, spark, sample_supplier_data, sample_nation_data, output_dir):
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(sample_supplier_data, sample_nation_data)

        assert result.duration_seconds > 0


class TestPipelineFailureHandling:
    """Tests for pipeline error handling."""

    def test_empty_supplier_data_fails_quality_gate(self, spark, sample_nation_data, output_dir):
        empty_data = {
            "s_suppkey": [],
            "s_name": [],
            "s_address": [],
            "s_nationkey": [],
            "s_phone": [],
            "s_acctbal": [],
        }
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(empty_data, sample_nation_data)

        assert result.success is False
        assert result.error is not None

    def test_failed_run_has_run_id(self, spark, sample_nation_data, output_dir):
        empty_data = {
            "s_suppkey": [],
            "s_name": [],
            "s_address": [],
            "s_nationkey": [],
            "s_phone": [],
            "s_acctbal": [],
        }
        pipeline = MedallionPipeline(spark, output_dir)
        result = pipeline.run(empty_data, sample_nation_data)

        assert result.run_id.startswith("run_")
