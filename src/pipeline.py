"""Main ETL Pipeline orchestrator."""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .extract import DataExtractor
from .transform import SupplierTransformer
from .validate import DataValidator
from .load import DataLoader
from .utils import PipelineTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    run_id: str
    input_rows: Dict[str, int]
    output_rows: int
    validation_passed: bool
    output_path: Optional[Path]
    duration_seconds: float
    error: Optional[str] = None


class SupplierETLPipeline:
    """
    ETL Pipeline for supplier data processing.

    Architecture:
    ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌────────┐
    │ EXTRACT │ -> │ TRANSFORM │ -> │ VALIDATE │ -> │  LOAD  │
    └─────────┘    └───────────┘    └──────────┘    └────────┘

    This pipeline:
    1. Extracts supplier and nation data
    2. Transforms by joining and cleaning
    3. Validates data quality
    4. Loads to destination
    """

    def __init__(
        self,
        output_path: Path,
        metrics_path: Optional[Path] = None,
        pipeline_name: str = "supplier_etl_pipeline"
    ):
        """
        Initialize the pipeline.

        Args:
            output_path: Path for output file
            metrics_path: Path for metrics CSV
            pipeline_name: Name of the pipeline
        """
        self.output_path = Path(output_path)
        self.metrics_path = metrics_path
        self.pipeline_name = pipeline_name

        # Initialize components
        self.transformer = SupplierTransformer()
        self.validator = self._setup_validator()
        self.tracker = None

    def _setup_validator(self) -> DataValidator:
        """Set up validation rules."""
        return (
            DataValidator()
            .add_uniqueness_check("supplier_name", is_critical=True)
            .add_distinct_count_check("supplier_nation", max_count=25, is_critical=True)
            .add_not_null_check("supplier_name", is_critical=True)
            .add_row_count_check(min_rows=1, is_critical=True)
        )

    def run(
        self,
        supplier_data: Dict[str, list],
        nation_data: Dict[str, list]
    ) -> PipelineResult:
        """
        Execute the ETL pipeline with dictionary data.

        Args:
            supplier_data: Supplier data as dictionary
            nation_data: Nation data as dictionary

        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()

        # Initialize tracker
        self.tracker = PipelineTracker(
            pipeline_name=self.pipeline_name,
            log_file=self.metrics_path
        )

        logger.info("=" * 60)
        logger.info(f"Starting {self.pipeline_name}")
        logger.info("=" * 60)

        try:
            # EXTRACT
            logger.info("EXTRACT: Loading data from sources")
            supplier_df = DataExtractor.from_dict(supplier_data)
            nation_df = DataExtractor.from_dict(nation_data)

            self.tracker.track_dataframes("extract", {
                "supplier": supplier_df,
                "nation": nation_df
            })

            input_rows = {
                "supplier": len(supplier_df),
                "nation": len(nation_df)
            }

            # TRANSFORM
            logger.info("TRANSFORM: Processing data")
            transformed_df = self.transformer.transform(supplier_df, nation_df)
            self.tracker.track_dataframe("transform", "output", transformed_df)

            # VALIDATE
            logger.info("VALIDATE: Running quality checks")
            validation_results = self.validator.validate(transformed_df)
            validation_passed = self.validator.all_passed()

            for result in validation_results:
                self.tracker.track(
                    "validate",
                    "quality_check",
                    result.rule_name,
                    "passed" if result.passed else "failed"
                )

            if not validation_passed:
                logger.error("Validation failed! Stopping pipeline.")
                self.tracker.save()

                duration = (datetime.now() - start_time).total_seconds()
                return PipelineResult(
                    success=False,
                    run_id=self.tracker.run_id,
                    input_rows=input_rows,
                    output_rows=0,
                    validation_passed=False,
                    output_path=None,
                    duration_seconds=duration,
                    error="Data validation failed"
                )

            # LOAD
            logger.info("LOAD: Saving to destination")
            output_path = DataLoader.to_csv(transformed_df, self.output_path)
            self.tracker.track("load", "output", "row_count", len(transformed_df))
            self.tracker.track("load", "output", "file_path", str(output_path))

            # Save metrics
            self.tracker.save()

            duration = (datetime.now() - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Output: {output_path}")
            logger.info("=" * 60)

            return PipelineResult(
                success=True,
                run_id=self.tracker.run_id,
                input_rows=input_rows,
                output_rows=len(transformed_df),
                validation_passed=True,
                output_path=output_path,
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

            if self.tracker:
                self.tracker.save()

            duration = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                success=False,
                run_id=self.tracker.run_id if self.tracker else "unknown",
                input_rows={},
                output_rows=0,
                validation_passed=False,
                output_path=None,
                duration_seconds=duration,
                error=str(e)
            )


def create_sample_data() -> tuple:
    """Create sample supplier and nation data for testing."""
    supplier_data = {
        "s_suppkey": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "s_name": [
            "Supplier#001", "Supplier#002", "Supplier#003", "Supplier#004", "Supplier#005",
            "Supplier#006", "Supplier#007", "Supplier#008", "Supplier#009", "Supplier#010"
        ],
        "s_address": [
            "Address1", "Address2", "Address3", "Address4", "Address5",
            "Address6", "Address7", "Address8", "Address9", "Address10"
        ],
        "s_nationkey": [17, 5, 24, 2, 19, 0, 23, 17, 10, 8],
        "s_phone": [
            "27-918-335-1736", "15-679-861-2259", "34-546-815-5376",
            "12-314-759-5682", "29-650-264-3518", "10-514-483-8935",
            "33-484-637-4851", "27-217-225-2336", "20-403-398-8662",
            "18-688-445-3302"
        ],
        "s_acctbal": [5755.94, 4032.68, 9694.28, 4656.24, 9653.32,
                      2963.17, 8142.56, 9862.18, 5733.73, 1048.88],
    }

    nation_data = {
        "n_nationkey": [0, 2, 5, 8, 10, 17, 19, 23, 24],
        "n_name": [
            "ALGERIA", "BRAZIL", "ETHIOPIA", "INDIA", "IRAN",
            "PERU", "ROMANIA", "UNITED KINGDOM", "UNITED STATES"
        ],
        "n_regionkey": [0, 1, 0, 2, 4, 1, 3, 3, 1],
    }

    return supplier_data, nation_data
