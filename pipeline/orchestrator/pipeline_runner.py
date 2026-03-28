"""End-to-end Medallion pipeline orchestrator.

Coordinates Bronze → Silver → Gold flow with quality gates between each layer,
metrics collection, and structured result reporting.

    ┌────────┐  DQ  ┌────────┐  DQ  ┌────────┐
    │ BRONZE │ ───> │ SILVER │ ───> │  GOLD  │
    └────────┘ Gate └────────┘ Gate └────────┘
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from pipeline.bronze.ingestion import BronzeIngestion, create_bronze_schemas
from pipeline.gold.aggregations import GoldAggregator
from pipeline.quality.checks import DataQualityChecker
from pipeline.silver.transformations import SilverTransformer
from pipeline.utils.metrics import PipelineMetrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Immutable result of a full pipeline run."""

    success: bool
    run_id: str
    input_rows: dict[str, int]
    silver_rows: int
    gold_tables: dict[str, int]
    duration_seconds: float
    quality_summary: dict = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


class MedallionPipeline:
    """Orchestrates the complete Bronze → Silver → Gold medallion ETL.

    Args:
        spark: SparkSession instance.
        output_dir: Root directory for pipeline outputs.
        pipeline_name: Logical name for logging and metrics.
    """

    def __init__(
        self,
        spark: SparkSession,
        output_dir: Path,
        pipeline_name: str = "supplier_medallion_etl",
    ):
        self.spark = spark
        self.output_dir = Path(output_dir)
        self.pipeline_name = pipeline_name

        self.bronze = None
        self.silver_transformer = SilverTransformer()
        self.gold_aggregator = GoldAggregator()
        self.metrics = PipelineMetrics(pipeline_name)

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        supplier_data: dict[str, list],
        nation_data: dict[str, list],
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            supplier_data: Raw supplier data as column-oriented dict.
            nation_data: Raw nation data as column-oriented dict.

        Returns:
            PipelineResult with execution details and metrics.
        """
        logger.info("=" * 70)
        logger.info("Starting %s [run_id=%s]", self.pipeline_name, self.metrics.run_id)
        logger.info("=" * 70)

        try:
            # ── BRONZE ────────────────────────────────────────────
            suppliers_bronze, nations_bronze = self._run_bronze(
                supplier_data, nation_data
            )
            input_rows = {
                "suppliers": suppliers_bronze.count(),
                "nations": nations_bronze.count(),
            }

            # ── Bronze quality gate ───────────────────────────────
            self._run_bronze_quality_gate(suppliers_bronze, nations_bronze)

            # ── SILVER ────────────────────────────────────────────
            silver_df = self._run_silver(suppliers_bronze, nations_bronze)

            # ── Silver quality gate ───────────────────────────────
            silver_qc = self._run_silver_quality_gate(silver_df)

            # ── GOLD ─────────────────────────────────────────────
            gold_tables = self._run_gold(silver_df)

            # ── Write outputs ─────────────────────────────────────
            output_paths = self._write_outputs(silver_df, gold_tables)

            logger.info("=" * 70)
            logger.info("Pipeline completed successfully in %.2fs", self.metrics.total_duration)
            logger.info("=" * 70)

            return PipelineResult(
                success=True,
                run_id=self.metrics.run_id,
                input_rows=input_rows,
                silver_rows=silver_df.count(),
                gold_tables={name: df.count() for name, df in gold_tables.items()},
                duration_seconds=self.metrics.total_duration,
                quality_summary=silver_qc.summary(),
                output_paths=output_paths,
            )

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return PipelineResult(
                success=False,
                run_id=self.metrics.run_id,
                input_rows={},
                silver_rows=0,
                gold_tables={},
                duration_seconds=self.metrics.total_duration,
                error=str(exc),
            )

    # ── Layer runners ─────────────────────────────────────────────────

    def _run_bronze(
        self,
        supplier_data: dict[str, list],
        nation_data: dict[str, list],
    ) -> tuple[DataFrame, DataFrame]:
        """Ingest raw data into Bronze layer."""
        schemas = create_bronze_schemas()

        supplier_bronze = BronzeIngestion(self.spark, "tpch_supplier")
        nation_bronze = BronzeIngestion(self.spark, "tpch_nation")

        with self.metrics.track_stage("bronze", "suppliers") as m:
            suppliers_df = supplier_bronze.ingest_from_dict(
                supplier_data, schema=schemas["supplier"]
            )
            m["df"] = suppliers_df

        with self.metrics.track_stage("bronze", "nations") as m:
            nations_df = nation_bronze.ingest_from_dict(
                nation_data, schema=schemas["nation"]
            )
            m["df"] = nations_df

        return suppliers_df, nations_df

    def _run_bronze_quality_gate(
        self,
        suppliers_df: DataFrame,
        nations_df: DataFrame,
    ) -> DataQualityChecker:
        """Validate Bronze data before Silver processing."""
        checker = (
            DataQualityChecker("bronze_gate")
            .expect_row_count_between(min_rows=1)
            .expect_column_not_null("_batch_id")
            .expect_column_not_null("_source")
        )
        checker.run(suppliers_df)
        if not checker.critical_passed():
            raise RuntimeError(
                f"Bronze quality gate failed: {checker.summary()}"
            )
        return checker

    def _run_silver(
        self,
        suppliers_bronze: DataFrame,
        nations_bronze: DataFrame,
    ) -> DataFrame:
        """Transform Bronze data into Silver layer."""
        with self.metrics.track_stage("silver", "supplier_conformed") as m:
            silver_df = self.silver_transformer.transform(
                suppliers_bronze, nations_bronze
            )
            m["df"] = silver_df
        return silver_df

    def _run_silver_quality_gate(self, silver_df: DataFrame) -> DataQualityChecker:
        """Validate Silver data before Gold aggregation."""
        checker = (
            DataQualityChecker("silver_gate")
            .expect_column_not_null("supplier_id")
            .expect_column_not_null("supplier_name")
            .expect_column_unique("supplier_id")
            .expect_row_count_between(min_rows=1)
            .expect_distinct_count_between("nation_name", min_count=1, max_count=25)
        )
        checker.run(silver_df)
        if not checker.critical_passed():
            raise RuntimeError(
                f"Silver quality gate failed: {checker.summary()}"
            )
        return checker

    def _run_gold(self, silver_df: DataFrame) -> dict[str, DataFrame]:
        """Build Gold aggregation tables."""
        gold_tables = {}
        with self.metrics.track_stage("gold", "supplier_by_nation") as m:
            gold_tables["supplier_by_nation"] = (
                self.gold_aggregator.build_supplier_by_nation(silver_df)
            )
            m["df"] = gold_tables["supplier_by_nation"]

        with self.metrics.track_stage("gold", "supplier_summary") as m:
            gold_tables["supplier_summary"] = (
                self.gold_aggregator.build_supplier_summary(silver_df)
            )
            m["df"] = gold_tables["supplier_summary"]

        return gold_tables

    def _write_outputs(
        self,
        silver_df: DataFrame,
        gold_tables: dict[str, DataFrame],
    ) -> dict[str, str]:
        """Write Silver and Gold outputs to Parquet."""
        paths: dict[str, str] = {}

        silver_path = self.output_dir / "silver" / "suppliers"
        silver_df.write.mode("overwrite").parquet(str(silver_path))
        paths["silver_suppliers"] = str(silver_path)
        logger.info("Wrote Silver → %s", silver_path)

        for name, df in gold_tables.items():
            gold_path = self.output_dir / "gold" / name
            df.write.mode("overwrite").parquet(str(gold_path))
            paths[f"gold_{name}"] = str(gold_path)
            logger.info("Wrote Gold %s → %s", name, gold_path)

        return paths
