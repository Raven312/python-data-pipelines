#!/usr/bin/env python3
"""
Python Data Pipelines - Main Entry Point

A modular ETL pipeline architecture designed with LLM assistance.

Usage:
    python main.py
    python main.py --output data/processed/output.csv
    python main.py --log-level DEBUG
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DEFAULT_PIPELINE_CONFIG
from src.pipeline import SupplierETLPipeline, create_sample_data
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Python Data Pipelines - ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: data/processed/supplier_data.csv)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Get configuration
    config = DEFAULT_PIPELINE_CONFIG

    # Set up logging
    logger = setup_logger(
        log_dir=config.logs_dir if not args.no_log_file else None,
        log_level=args.log_level,
        log_to_file=not args.no_log_file,
        log_to_console=True
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = config.processed_data_dir / "supplier_data.csv"

    # Ensure directories exist
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Create sample data
    supplier_data, nation_data = create_sample_data()

    # Create and run pipeline
    pipeline = SupplierETLPipeline(
        output_path=output_path,
        metrics_path=config.logs_dir / "pipeline_metrics.csv",
        pipeline_name=config.pipeline_name
    )

    result = pipeline.run(supplier_data, nation_data)

    # Print summary
    print("\n" + "=" * 60)
    if result.success:
        print("Pipeline completed successfully!")
        print(f"  Run ID:       {result.run_id}")
        print(f"  Input rows:   {result.input_rows}")
        print(f"  Output rows:  {result.output_rows}")
        print(f"  Duration:     {result.duration_seconds:.2f}s")
        print(f"  Output file:  {result.output_path}")
        sys.exit(0)
    else:
        print(f"Pipeline failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
