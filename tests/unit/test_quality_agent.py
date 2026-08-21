"""Unit tests for the Data Quality Agent."""

import pytest
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from agents.quality.quality_agent import (
    AnomalyLevel,
    DataQualityAgent,
    QualityProfile,
)


@pytest.fixture
def clean_df(spark):
    return spark.createDataFrame(
        [(1, "Alice", 100.0), (2, "Bob", 200.0), (3, "Charlie", 300.0)],
        schema=["id", "name", "balance"],
    )


@pytest.fixture
def dirty_df(spark):
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("name", StringType()),
        StructField("balance", DoubleType()),
    ])
    return spark.createDataFrame(
        [(1, None, 100.0), (2, None, None), (3, None, None)],
        schema=schema,
    )


@pytest.fixture
def agent_with_profile():
    profile = QualityProfile(
        dataset_name="test",
        expected_row_count_mean=3.0,
        expected_row_count_stddev=1.0,
        expected_null_rate_mean=0.0,
        expected_null_rate_stddev=0.01,
    )
    return DataQualityAgent(
        dataset_name="test",
        key_column="id",
        critical_columns=["name", "balance"],
        profile=profile,
    )


class TestQualityAnalysis:

    def test_clean_data_passes(self, clean_df, agent_with_profile):
        metrics = agent_with_profile.analyze(clean_df)
        levels = [m.level for m in metrics]
        assert AnomalyLevel.CRITICAL not in levels

    def test_dirty_data_detects_anomaly(self, dirty_df, agent_with_profile):
        metrics = agent_with_profile.analyze(dirty_df)
        critical = [m for m in metrics if m.level == AnomalyLevel.CRITICAL]
        assert len(critical) > 0

    def test_null_rate_detected(self, dirty_df, agent_with_profile):
        metrics = agent_with_profile.analyze(dirty_df)
        null_metrics = [m for m in metrics if "null_rate" in m.metric_name]
        assert any(m.current_value > 0 for m in null_metrics)

    def test_z_score_computed(self, clean_df, agent_with_profile):
        metrics = agent_with_profile.analyze(clean_df)
        for m in metrics:
            assert isinstance(m.z_score, float)


class TestQualityDecisions:

    def test_pass_on_clean(self, clean_df, agent_with_profile):
        metrics = agent_with_profile.analyze(clean_df)
        decision = agent_with_profile.decide(metrics)
        assert decision["action"] in ("pass", "warn")

    def test_quarantine_on_critical(self, dirty_df, agent_with_profile):
        metrics = agent_with_profile.analyze(dirty_df)
        decision = agent_with_profile.decide(metrics)
        assert decision["action"] in ("quarantine", "escalate")

    def test_escalation_after_consecutive(self, dirty_df, agent_with_profile):
        for _ in range(4):
            metrics = agent_with_profile.analyze(dirty_df)
        decision = agent_with_profile.decide(metrics)
        assert decision["action"] == "escalate"


class TestQualityProfile:

    def test_profile_updates(self, clean_df, agent_with_profile):
        agent_with_profile.analyze(clean_df)
        assert len(agent_with_profile.profile.observations) >= 1

    def test_summary(self, clean_df, agent_with_profile):
        agent_with_profile.analyze(clean_df)
        summary = agent_with_profile.get_summary()
        assert summary["dataset"] == "test"
        assert summary["total_observations"] >= 1
