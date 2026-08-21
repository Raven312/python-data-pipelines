"""Unit tests for the Schema Drift Agent."""

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, DoubleType

from agents.schema.schema_agent import DriftType, SchemaDriftAgent


@pytest.fixture
def base_schema():
    return StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("value", DoubleType(), True),
    ])


@pytest.fixture
def agent(base_schema):
    return SchemaDriftAgent("test_dataset", base_schema)


class TestDriftDetection:

    def test_no_drift(self, spark, agent, base_schema):
        df = spark.createDataFrame([(1, "A", 1.0)], schema=base_schema)
        drifts = agent.detect(df)
        assert len(drifts) == 0

    def test_additive_column(self, spark, agent):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("value", DoubleType()),
            StructField("new_col", StringType()),
        ])
        df = spark.createDataFrame([(1, "A", 1.0, "extra")], schema=schema)
        drifts = agent.detect(df)
        additive = [d for d in drifts if d.drift_type == DriftType.ADDITIVE]
        assert len(additive) == 1
        assert additive[0].column_name == "new_col"

    def test_dropped_column(self, spark, agent):
        df = spark.createDataFrame([(1, "A")], schema=["id", "name"])
        drifts = agent.detect(df)
        dropped = [d for d in drifts if d.drift_type == DriftType.COLUMN_DROPPED]
        assert len(dropped) == 1
        assert dropped[0].column_name == "value"

    def test_type_change(self, spark, agent):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("value", StringType()),  # was DoubleType
        ])
        df = spark.createDataFrame([(1, "A", "1.0")], schema=schema)
        drifts = agent.detect(df)
        type_changes = [d for d in drifts if d.drift_type == DriftType.TYPE_CHANGE]
        assert len(type_changes) == 1
        assert type_changes[0].column_name == "value"

    def test_audit_columns_ignored(self, spark, agent):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("value", DoubleType()),
            StructField("_ingested_at", StringType()),
        ])
        df = spark.createDataFrame([(1, "A", 1.0, "2024-01-01")], schema=schema)
        drifts = agent.detect(df)
        assert len(drifts) == 0


class TestDriftDecisions:

    def test_no_drift_accepted(self, agent):
        decision = agent.decide([])
        assert decision["action"] == "accept"

    def test_additive_evolves(self, spark, agent):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
            StructField("value", DoubleType()),
            StructField("region", StringType()),
        ])
        df = spark.createDataFrame([(1, "A", 1.0, "US")], schema=schema)
        drifts = agent.detect(df)
        decision = agent.decide(drifts)
        assert decision["action"] == "evolve"

    def test_breaking_rejected(self, spark, agent):
        df = spark.createDataFrame([(1, "A")], schema=["id", "name"])
        drifts = agent.detect(df)
        decision = agent.decide(drifts)
        assert decision["action"] == "reject"

    def test_schema_evolution(self, agent):
        agent.evolve_schema([{"name": "region", "type": "StringType"}])
        assert "region" in [f.name for f in agent.expected_schema.fields]


class TestSummary:

    def test_summary_no_drifts(self, agent):
        summary = agent.get_summary()
        assert summary["dataset"] == "test_dataset"
        assert summary["expected_columns"] == 3

    def test_summary_with_drifts(self, spark, agent):
        df = spark.createDataFrame([(1, "A")], schema=["id", "name"])
        agent.detect(df)
        summary = agent.get_summary()
        assert summary["total_drifts_detected"] > 0
