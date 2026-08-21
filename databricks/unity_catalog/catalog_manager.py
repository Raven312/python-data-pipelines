"""Unity Catalog management — catalog, schema, and table operations.

Unity Catalog provides a three-level namespace (catalog.schema.table)
for governing all data assets. This module handles:
  - Catalog and schema provisioning
  - Table creation with Delta Lake properties
  - Access control (GRANT/REVOKE)
  - Data lineage queries
  - Schema evolution management

AE Pathway Coverage:
  - Unity Catalog fundamentals
  - Data governance and access control
  - Three-level namespace management
  - Delta Lake table properties
"""

import logging
from dataclasses import dataclass, field

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


@dataclass
class CatalogConfig:
    """Unity Catalog configuration for a medallion pipeline."""

    catalog_name: str = "supply_chain"
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    staging_schema: str = "staging"

    @property
    def bronze_prefix(self) -> str:
        return f"{self.catalog_name}.{self.bronze_schema}"

    @property
    def silver_prefix(self) -> str:
        return f"{self.catalog_name}.{self.silver_schema}"

    @property
    def gold_prefix(self) -> str:
        return f"{self.catalog_name}.{self.gold_schema}"


class UnityCatalogManager:
    """Manages Unity Catalog resources for the medallion pipeline.

    Handles the full lifecycle of catalogs, schemas, and managed tables
    with appropriate permissions and properties.
    """

    def __init__(self, spark: SparkSession, config: CatalogConfig):
        self.spark = spark
        self.config = config

    def setup_catalog(self) -> None:
        """Provision the complete catalog structure for the medallion pipeline."""
        self._create_catalog()
        for schema in [
            self.config.bronze_schema,
            self.config.silver_schema,
            self.config.gold_schema,
            self.config.staging_schema,
        ]:
            self._create_schema(schema)
        logger.info("Catalog '%s' fully provisioned", self.config.catalog_name)

    def _create_catalog(self) -> None:
        self.spark.sql(
            f"CREATE CATALOG IF NOT EXISTS {self.config.catalog_name}"
        )
        self.spark.sql(
            f"ALTER CATALOG {self.config.catalog_name} "
            f"SET DBPROPERTIES ('purpose' = 'supply_chain_analytics', 'tier' = 'production')"
        )
        logger.info("Catalog '%s' created", self.config.catalog_name)

    def _create_schema(self, schema_name: str) -> None:
        fqn = f"{self.config.catalog_name}.{schema_name}"
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn}")
        self.spark.sql(
            f"ALTER SCHEMA {fqn} SET DBPROPERTIES ('layer' = '{schema_name}')"
        )
        logger.info("Schema '%s' created", fqn)

    def create_bronze_table(self, table_name: str, ddl_columns: str) -> str:
        """Create a Bronze Delta table with audit columns and ingestion properties."""
        fqn = f"{self.config.bronze_prefix}.{table_name}"
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                {ddl_columns},
                _ingested_at TIMESTAMP,
                _source STRING,
                _batch_id STRING
            )
            USING DELTA
            TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact' = 'true',
                'quality' = 'bronze'
            )
        """)
        logger.info("Bronze table '%s' created", fqn)
        return fqn

    def create_silver_table(self, table_name: str, ddl_columns: str) -> str:
        """Create a Silver Delta table with SCD-2 columns and optimization."""
        fqn = f"{self.config.silver_prefix}.{table_name}"
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                {ddl_columns},
                _valid_from TIMESTAMP,
                _valid_to TIMESTAMP,
                _is_current BOOLEAN
            )
            USING DELTA
            TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact' = 'true',
                'delta.enableChangeDataFeed' = 'true',
                'quality' = 'silver'
            )
        """)
        logger.info("Silver table '%s' created", fqn)
        return fqn

    def create_gold_table(self, table_name: str, ddl_columns: str) -> str:
        """Create a Gold Delta table optimized for query performance."""
        fqn = f"{self.config.gold_prefix}.{table_name}"
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                {ddl_columns},
                _computed_at TIMESTAMP
            )
            USING DELTA
            TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'quality' = 'gold'
            )
        """)
        logger.info("Gold table '%s' created", fqn)
        return fqn

    def grant_read_access(self, schema: str, principal: str) -> None:
        """Grant SELECT access on a schema to a principal (user/group/service principal)."""
        fqn = f"{self.config.catalog_name}.{schema}"
        self.spark.sql(f"GRANT USE SCHEMA ON SCHEMA {fqn} TO `{principal}`")
        self.spark.sql(f"GRANT SELECT ON SCHEMA {fqn} TO `{principal}`")
        logger.info("Granted read access on '%s' to '%s'", fqn, principal)

    def grant_write_access(self, schema: str, principal: str) -> None:
        """Grant full CRUD access on a schema to a principal."""
        fqn = f"{self.config.catalog_name}.{schema}"
        self.spark.sql(f"GRANT USE SCHEMA ON SCHEMA {fqn} TO `{principal}`")
        self.spark.sql(f"GRANT SELECT, MODIFY, CREATE TABLE ON SCHEMA {fqn} TO `{principal}`")
        logger.info("Granted write access on '%s' to '%s'", fqn, principal)

    def get_table_lineage(self, table_name: str) -> list[dict]:
        """Query Unity Catalog lineage for a table's upstream dependencies.

        Returns a list of upstream tables/columns that feed into the target.
        Requires Unity Catalog lineage feature (available on Premium+).
        """
        try:
            lineage_df = self.spark.sql(f"""
                SELECT *
                FROM system.access.table_lineage
                WHERE target_table_full_name = '{table_name}'
            """)
            return [row.asDict() for row in lineage_df.collect()]
        except Exception as exc:
            logger.warning("Lineage query failed (requires Premium tier): %s", exc)
            return []

    def get_table_history(self, table_fqn: str, limit: int = 10) -> list[dict]:
        """Get Delta Lake table history (time travel metadata)."""
        history_df = self.spark.sql(f"DESCRIBE HISTORY {table_fqn} LIMIT {limit}")
        return [row.asDict() for row in history_df.collect()]
