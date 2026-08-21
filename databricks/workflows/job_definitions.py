"""Databricks Workflows — Job definitions for pipeline orchestration.

Defines the Databricks Workflows job specification as Python dicts
that can be deployed via the Databricks SDK or REST API. Each job
represents a stage of the medallion pipeline with proper dependency
ordering, retry policies, and alerting.

AE Pathway Coverage:
  - Databricks Workflows fundamentals
  - Multi-task jobs with dependencies
  - DLT pipeline tasks
  - Notebook tasks
  - Cluster policies and configurations
  - Schedule and trigger configurations
  - Email/webhook notifications
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Databricks cluster configuration for workflow tasks."""

    spark_version: str = "14.3.x-scala2.12"
    node_type_id: str = "i3.xlarge"
    num_workers: int = 2
    autoscale_min: int = 1
    autoscale_max: int = 4
    spark_conf: dict = field(default_factory=lambda: {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.databricks.delta.optimizeWrite.enabled": "true",
        "spark.databricks.delta.autoCompact.enabled": "true",
    })

    def to_dict(self) -> dict:
        return {
            "spark_version": self.spark_version,
            "node_type_id": self.node_type_id,
            "autoscale": {
                "min_workers": self.autoscale_min,
                "max_workers": self.autoscale_max,
            },
            "spark_conf": self.spark_conf,
        }


def build_medallion_workflow(
    catalog: str = "supply_chain",
    dlt_pipeline_id: Optional[str] = None,
    notification_emails: Optional[list[str]] = None,
    schedule_cron: str = "0 0 6 * * ?",
    timezone: str = "UTC",
) -> dict:
    """Build a complete Databricks Workflows job definition.

    Creates a multi-task job DAG:
      1. setup_catalog      → Provision Unity Catalog resources
      2. bronze_ingestion   → DLT pipeline (Bronze layer)
      3. bronze_quality_gate → Quality checks on Bronze data
      4. silver_transform   → DLT pipeline (Silver layer)
      5. silver_quality_gate → Quality checks on Silver data
      6. gold_aggregations  → DLT pipeline (Gold layer)
      7. optimize_tables    → OPTIMIZE + ZORDER on key tables
      8. notify_completion  → Send success notification

    Args:
        catalog: Unity Catalog name.
        dlt_pipeline_id: DLT pipeline ID (if using managed DLT).
        notification_emails: Emails for failure alerts.
        schedule_cron: Quartz cron expression for scheduling.
        timezone: Timezone for the schedule.

    Returns:
        Job definition dict (Databricks REST API format).
    """
    cluster = ClusterConfig()
    emails = notification_emails or []

    tasks = [
        {
            "task_key": "setup_catalog",
            "description": "Provision Unity Catalog schemas and tables",
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/01_setup_catalog",
                "base_parameters": {"catalog": catalog},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 300,
            "max_retries": 1,
        },
        {
            "task_key": "bronze_ingestion",
            "description": "Ingest raw data into Bronze layer via Auto Loader",
            "depends_on": [{"task_key": "setup_catalog"}],
            **(
                {"pipeline_task": {"pipeline_id": dlt_pipeline_id, "full_refresh": False}}
                if dlt_pipeline_id
                else {
                    "notebook_task": {
                        "notebook_path": "/Repos/etl/databricks/notebooks/02_bronze_ingestion",
                        "base_parameters": {"catalog": catalog},
                    },
                    "new_cluster": cluster.to_dict(),
                }
            ),
            "timeout_seconds": 1800,
            "max_retries": 2,
            "retry_on_timeout": True,
        },
        {
            "task_key": "bronze_quality_gate",
            "description": "Run quality checks on Bronze data",
            "depends_on": [{"task_key": "bronze_ingestion"}],
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/03_quality_gate",
                "base_parameters": {"catalog": catalog, "layer": "bronze"},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 600,
            "max_retries": 0,
        },
        {
            "task_key": "silver_transform",
            "description": "Transform Bronze data into Silver (cleanse, join, SCD-2)",
            "depends_on": [{"task_key": "bronze_quality_gate"}],
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/04_silver_transform",
                "base_parameters": {"catalog": catalog},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 1800,
            "max_retries": 2,
        },
        {
            "task_key": "silver_quality_gate",
            "description": "Run quality checks on Silver data",
            "depends_on": [{"task_key": "silver_transform"}],
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/05_quality_gate",
                "base_parameters": {"catalog": catalog, "layer": "silver"},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 600,
            "max_retries": 0,
        },
        {
            "task_key": "gold_aggregations",
            "description": "Build Gold aggregation tables for dashboards",
            "depends_on": [{"task_key": "silver_quality_gate"}],
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/06_gold_aggregations",
                "base_parameters": {"catalog": catalog},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 1200,
            "max_retries": 1,
        },
        {
            "task_key": "optimize_tables",
            "description": "OPTIMIZE and ZORDER key Delta tables",
            "depends_on": [{"task_key": "gold_aggregations"}],
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/07_optimize",
                "base_parameters": {"catalog": catalog},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 1200,
            "max_retries": 1,
        },
        {
            "task_key": "notify_completion",
            "description": "Send pipeline completion notification",
            "depends_on": [{"task_key": "optimize_tables"}],
            "notebook_task": {
                "notebook_path": "/Repos/etl/databricks/notebooks/08_notify",
                "base_parameters": {"catalog": catalog, "status": "SUCCESS"},
            },
            "new_cluster": cluster.to_dict(),
            "timeout_seconds": 120,
        },
    ]

    job_definition = {
        "name": f"medallion_etl_{catalog}",
        "description": "Medallion ETL pipeline: Bronze → Silver → Gold with quality gates",
        "tasks": tasks,
        "schedule": {
            "quartz_cron_expression": schedule_cron,
            "timezone_id": timezone,
            "pause_status": "UNPAUSED",
        },
        "email_notifications": {
            "on_failure": emails,
            "on_success": [],
            "no_alert_for_skipped_runs": True,
        },
        "webhook_notifications": {},
        "max_concurrent_runs": 1,
        "timeout_seconds": 7200,
        "tags": {
            "team": "data-engineering",
            "pipeline": "medallion",
            "catalog": catalog,
        },
        "format": "MULTI_TASK",
    }

    logger.info("Built workflow '%s' with %d tasks", job_definition["name"], len(tasks))
    return job_definition


def export_job_json(job_definition: dict, output_path: str) -> str:
    """Export a job definition as JSON for deployment."""
    with open(output_path, "w") as f:
        json.dump(job_definition, f, indent=2)
    logger.info("Exported job definition to %s", output_path)
    return output_path
