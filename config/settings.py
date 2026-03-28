"""Pipeline configuration settings with multi-environment support.

Supports local, databricks, and test environments with appropriate
defaults for each. Configuration can be overridden via environment
variables or explicit parameter passing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    pipeline_name: str = "supplier_medallion_etl"
    environment: str = field(
        default_factory=lambda: os.getenv("PIPELINE_ENV", "local")
    )

    # Spark
    spark_app_name: str = "MedallionETL"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

    # Quality gates
    max_distinct_nations: int = 25
    require_unique_supplier_names: bool = True
    min_row_count: int = 1

    # ── Derived paths ─────────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"


@dataclass
class SupplierDataConfig:
    """Schema configuration for supplier TPC-H data."""

    # Source columns
    supplier_key_col: str = "s_suppkey"
    supplier_name_col: str = "s_name"
    supplier_address_col: str = "s_address"
    supplier_nation_key_col: str = "s_nationkey"
    supplier_phone_col: str = "s_phone"
    supplier_acctbal_col: str = "s_acctbal"

    # Nation columns
    nation_key_col: str = "n_nationkey"
    nation_name_col: str = "n_name"
    nation_region_key_col: str = "n_regionkey"


# ── Default instances ─────────────────────────────────────────────────

DEFAULT_PIPELINE_CONFIG = PipelineConfig()
DEFAULT_DATA_CONFIG = SupplierDataConfig()
