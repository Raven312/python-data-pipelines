"""
Enterprise ETL Pipeline - PySpark Medallion Architecture

A FAANG-grade data engineering framework implementing the Bronze → Silver → Gold
medallion pattern with comprehensive data quality checks, Delta Lake integration,
and Databricks-native patterns.

Architecture:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  BRONZE  │ -> │  SILVER  │ -> │   GOLD   │
    │  (Raw)   │    │ (Clean)  │    │  (Biz)   │
    └──────────┘    └──────────┘    └──────────┘
         │               │               │
         └───── Data Quality Gates ──────┘
"""

from pipeline.utils.spark_factory import SparkSessionFactory
from pipeline.orchestrator.pipeline_runner import MedallionPipeline

__all__ = ["SparkSessionFactory", "MedallionPipeline"]
