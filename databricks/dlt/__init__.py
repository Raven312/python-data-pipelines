"""Delta Live Tables (DLT) — Declarative Medallion Pipeline.

Delta Live Tables is Databricks' declarative ETL framework. Instead of
writing imperative Spark code, you declare WHAT each table should contain
and DLT handles orchestration, dependency resolution, error handling,
and data quality enforcement.

This module implements the full Bronze → Silver → Gold medallion pattern
as DLT table definitions with built-in quality expectations.

AE Pathway Coverage:
  - Delta Live Tables fundamentals
  - Expectations (data quality constraints)
  - Medallion architecture in DLT
  - Schema enforcement and evolution
"""
