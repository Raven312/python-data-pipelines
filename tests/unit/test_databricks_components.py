"""Unit tests for Databricks components that can run locally."""

import json
import pytest

from databricks.sql.analytics_queries import get_query
from databricks.workflows.job_definitions import (
    ClusterConfig,
    build_medallion_workflow,
)


class TestAnalyticsQueries:

    def test_get_known_query(self):
        sql = get_query("supplier_kpi_dashboard", catalog="test_catalog")
        assert "test_catalog" in sql
        assert "total_suppliers" in sql

    def test_get_nation_distribution(self):
        sql = get_query("supplier_nation_distribution")
        assert "supply_chain" in sql
        assert "nation_rank" in sql

    def test_get_health_monitor(self):
        sql = get_query("pipeline_health_monitor")
        assert "freshness_status" in sql

    def test_get_quality_audit(self):
        sql = get_query("data_quality_audit")
        assert "null_keys" in sql

    def test_unknown_query_raises(self):
        with pytest.raises(ValueError, match="Unknown query"):
            get_query("nonexistent_query")


class TestClusterConfig:

    def test_default_config(self):
        config = ClusterConfig()
        d = config.to_dict()
        assert "spark_version" in d
        assert d["autoscale"]["min_workers"] == 1

    def test_custom_config(self):
        config = ClusterConfig(num_workers=8, node_type_id="r5.2xlarge")
        d = config.to_dict()
        assert d["node_type_id"] == "r5.2xlarge"


class TestWorkflowDefinitions:

    def test_build_workflow(self):
        job = build_medallion_workflow(catalog="test_catalog")
        assert job["name"] == "medallion_etl_test_catalog"
        assert len(job["tasks"]) == 8

    def test_task_dependencies(self):
        job = build_medallion_workflow()
        tasks = {t["task_key"]: t for t in job["tasks"]}
        assert "depends_on" not in tasks["setup_catalog"]
        assert tasks["bronze_ingestion"]["depends_on"][0]["task_key"] == "setup_catalog"
        assert tasks["silver_transform"]["depends_on"][0]["task_key"] == "bronze_quality_gate"
        assert tasks["gold_aggregations"]["depends_on"][0]["task_key"] == "silver_quality_gate"

    def test_schedule_config(self):
        job = build_medallion_workflow(schedule_cron="0 0 8 * * ?", timezone="America/New_York")
        assert job["schedule"]["quartz_cron_expression"] == "0 0 8 * * ?"
        assert job["schedule"]["timezone_id"] == "America/New_York"

    def test_notification_config(self):
        job = build_medallion_workflow(notification_emails=["team@example.com"])
        assert "team@example.com" in job["email_notifications"]["on_failure"]

    def test_dlt_pipeline_task(self):
        job = build_medallion_workflow(dlt_pipeline_id="abc123")
        bronze_task = next(t for t in job["tasks"] if t["task_key"] == "bronze_ingestion")
        assert "pipeline_task" in bronze_task
        assert bronze_task["pipeline_task"]["pipeline_id"] == "abc123"

    def test_serializable(self):
        job = build_medallion_workflow()
        json_str = json.dumps(job)
        assert len(json_str) > 100
        parsed = json.loads(json_str)
        assert parsed["name"] == job["name"]

    def test_tags(self):
        job = build_medallion_workflow(catalog="prod")
        assert job["tags"]["catalog"] == "prod"
        assert job["tags"]["pipeline"] == "medallion"
