#!/usr/bin/env python3
"""
Enterprise Medallion ETL Pipeline — Main Entry Point

PySpark-based data engineering pipeline implementing the Bronze → Silver → Gold
medallion architecture with data quality gates between each layer.

Usage:
    python main.py
    python main.py --env local --log-level DEBUG
    python main.py --output data/output
"""

import argparse
import sys
from pathlib import Path

from config import DEFAULT_PIPELINE_CONFIG
from pipeline.orchestrator.pipeline_runner import MedallionPipeline
from pipeline.utils.spark_factory import SparkSessionFactory
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enterprise Medallion ETL Pipeline (PySpark)",
    )
    parser.add_argument(
        "--env",
        choices=["local", "databricks", "test"],
        default="local",
        help="Execution environment (default: local)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--no-log-file", action="store_true")
    return parser.parse_args()


def create_sample_data() -> tuple[dict, dict]:
    """TPC-H inspired sample supplier and nation data."""
    supplier_data = {
        "s_suppkey": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "s_name": [
            "Supplier#001", "Supplier#002", "Supplier#003", "Supplier#004",
            "Supplier#005", "Supplier#006", "Supplier#007", "Supplier#008",
            "Supplier#009", "Supplier#010",
        ],
        "s_address": [
            "Address1", "Address2", "Address3", "Address4", "Address5",
            "Address6", "Address7", "Address8", "Address9", "Address10",
        ],
        "s_nationkey": [17, 5, 24, 2, 19, 0, 23, 17, 10, 8],
        "s_phone": [
            "27-918-335-1736", "15-679-861-2259", "34-546-815-5376",
            "12-314-759-5682", "29-650-264-3518", "10-514-483-8935",
            "33-484-637-4851", "27-217-225-2336", "20-403-398-8662",
            "18-688-445-3302",
        ],
        "s_acctbal": [
            5755.94, 4032.68, 9694.28, 4656.24, 9653.32,
            2963.17, 8142.56, 9862.18, 5733.73, 1048.88,
        ],
    }

    nation_data = {
        "n_nationkey": [0, 2, 5, 8, 10, 17, 19, 23, 24],
        "n_name": [
            "ALGERIA", "BRAZIL", "ETHIOPIA", "INDIA", "IRAN",
            "PERU", "ROMANIA", "UNITED KINGDOM", "UNITED STATES",
        ],
        "n_regionkey": [0, 1, 0, 2, 4, 1, 3, 3, 1],
    }
    return supplier_data, nation_data


def main():
    args = parse_args()
    config = DEFAULT_PIPELINE_CONFIG

    setup_logger(
        log_dir=config.logs_dir if not args.no_log_file else None,
        log_level=args.log_level,
        log_to_file=not args.no_log_file,
        log_to_console=True,
    )

    output_dir = args.output or config.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSessionFactory.get_or_create(
        app_name=config.spark_app_name,
        environment=args.env,
    )

    try:
        supplier_data, nation_data = create_sample_data()

        pipeline = MedallionPipeline(
            spark=spark,
            output_dir=output_dir,
            pipeline_name=config.pipeline_name,
        )
        result = pipeline.run(supplier_data, nation_data)

        print("\n" + "=" * 70)
        if result.success:
            print("Pipeline completed successfully!")
            print(f"  Run ID:        {result.run_id}")
            print(f"  Input rows:    {result.input_rows}")
            print(f"  Silver rows:   {result.silver_rows}")
            print(f"  Gold tables:   {result.gold_tables}")
            print(f"  Duration:      {result.duration_seconds:.2f}s")
            print(f"  Quality:       {result.quality_summary}")
            print(f"  Outputs:       {list(result.output_paths.keys())}")
            sys.exit(0)
        else:
            print(f"Pipeline FAILED: {result.error}")
            sys.exit(1)
    finally:
        SparkSessionFactory.stop()


if __name__ == "__main__":
    main()
