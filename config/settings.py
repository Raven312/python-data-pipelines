"""Pipeline configuration settings."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PipelineConfig:
    """Configuration for the ETL pipeline."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    # Pipeline settings
    pipeline_name: str = "supplier_etl_pipeline"

    # Validation rules
    max_distinct_nations: int = 25
    require_unique_supplier_names: bool = True

    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True


@dataclass
class SupplierDataConfig:
    """Configuration for supplier data schema."""

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

    # Output columns
    output_columns: Dict[str, str] = field(default_factory=lambda: {
        "supplier_name": "supplier_name",
        "supplier_phone_number": "supplier_phone_number",
        "supplier_nation": "supplier_nation"
    })


# Default configurations
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
DEFAULT_DATA_CONFIG = SupplierDataConfig()
