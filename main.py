#!/usr/bin/env python3
"""
Enterprise Medallion ETL Pipeline — Main Entry Point

PySpark-based data engineering pipeline implementing the Bronze → Silver → Gold
medallion architecture with data quality gates between each layer.

Usage:
    python main.py
    python main.py --env local --log-level DEBUG
    python main.py --output data/output

Agentic modes (Harness + Loop):
    python main.py --mode agentic --auto-approve
    python main.py --mode agentic --inject-fault transient --auto-approve
    python main.py --mode agentic --inject-fault empty_batch
    python main.py --mode agentic --policy claude      # needs ANTHROPIC_API_KEY
    python main.py --mode monitor --cycles 3 --interval 5 --auto-approve
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
    parser.add_argument(
        "--mode",
        choices=["batch", "agentic", "monitor"],
        default="batch",
        help="batch: classic run · agentic: agent loop drives the stages · "
             "monitor: recurring SLA loop with agentic remediation",
    )
    parser.add_argument("--policy", choices=["rules", "claude"], default="rules")
    parser.add_argument(
        "--inject-fault",
        choices=sorted(FAULT_SCENARIOS),
        default=None,
        help="Replay an incident to watch the agent recover or escalate",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve the gated publish step without prompting",
    )
    parser.add_argument("--cycles", type=int, default=3, help="Monitor ticks (monitor mode)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between ticks")
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


def _drop_key(data: dict, key: str) -> dict:
    return {k: v for k, v in data.items() if k != key}


# Each scenario rewrites the sample batch and/or scripts stage faults.
FAULT_SCENARIOS = {
    "transient": lambda s, n: (s, n, {"run_silver": [RuntimeError("503 Service Unavailable: throttled")]}),
    "empty_batch": lambda s, n: ({k: [] for k in s}, n, {}),
    "additive_column": lambda s, n: ({**s, "s_comment": ["new field"] * len(s["s_suppkey"])}, n, {}),
    "dropped_column": lambda s, n: (_drop_key(s, "s_phone"), n, {}),
    "unknown_error": lambda s, n: (s, n, {"run_gold": [RuntimeError("unexpected executor state")]}),
}


def prompt_approver(tool_name: str, args: dict) -> bool:
    """Human-in-the-loop gate for destructive tools."""
    if not sys.stdin.isatty():
        return False
    answer = input(f"\nAgent requests '{tool_name}' {args or ''}. Approve? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def run_agentic(args, spark, output_dir, config) -> int:
    from agents.harness import approve_all
    from agents.harness.pipeline_toolkit import FaultInjector
    from agents.loop.operator import run_agentic_pipeline

    supplier_data, nation_data = create_sample_data()
    faults = {}
    if args.inject_fault:
        supplier_data, nation_data, faults = FAULT_SCENARIOS[args.inject_fault](supplier_data, nation_data)

    result, toolkit = run_agentic_pipeline(
        spark, output_dir, supplier_data, nation_data,
        policy=args.policy,
        approver=approve_all if args.auto_approve else prompt_approver,
        faults=FaultInjector(faults),
        audit_path=config.logs_dir / "agent_audit.jsonl",
        alert_sink=lambda msg: print(f"\n[ALERT] {msg}"),
    )

    print("\n" + "=" * 70)
    print(f"Agent run ({args.policy} policy){f' — fault: {args.inject_fault}' if args.inject_fault else ''}")
    print("=" * 70)
    print("\n".join(result.transcript()))
    print("\nIncident report:")
    for key, value in result.incident_report().items():
        print(f"  {key:<18} {value}")
    print(f"\nAudit: {result.audit}")
    return 0 if result.succeeded else 1


def run_monitor(args, spark, output_dir) -> int:
    from agents.harness import approve_all
    from agents.loop.monitor import MonitoringLoop
    from agents.loop.operator import make_remediation
    from agents.sla.sla_agent import SLADefinition, SLAMonitorAgent

    sla = SLAMonitorAgent(SLADefinition(name="supplier_gold", max_freshness_minutes=60))
    remediate = make_remediation(
        spark, output_dir,
        load_batch=create_sample_data,
        policy=args.policy,
        approver=approve_all if args.auto_approve else prompt_approver,
    )
    monitor = MonitoringLoop(
        sla, remediate,
        interval_seconds=args.interval,
        alert_sink=lambda msg: print(f"\n[ALERT] {msg}"),
    )
    for tick in monitor.run(cycles=args.cycles):
        run = f" → remediation {tick.run.status}" if tick.run else ""
        print(f"cycle {tick.cycle}: SLA {tick.sla_action} {tick.breached_metrics or ''}{run}")
    return 0 if not monitor.escalated else 1


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
        if args.mode == "agentic":
            sys.exit(run_agentic(args, spark, output_dir, config))
        if args.mode == "monitor":
            sys.exit(run_monitor(args, spark, output_dir))

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
