"""
Agentic Workflow Framework for Data Pipelines

Autonomous agents that monitor, heal, and optimize the medallion ETL
pipeline. Each agent operates independently with its own decision loop,
communicating through a shared event bus.

Agents:
  - PipelineOrchestrationAgent: Monitors runs, retries failures, manages state
  - DataQualityAgent: Detects anomalies, quarantines bad data, auto-remediates
  - SchemaDriftAgent: Detects upstream schema changes and adapts the pipeline
  - SLAMonitorAgent: Tracks pipeline freshness, latency, and SLA compliance

Execution layers:
  - agents.harness: bounded tools, guardrails, hooks and audit around every
    action an agent takes on the pipeline
  - agents.loop: the observe → decide → act loop that drives the pipeline,
    a rule-based or Claude policy, and a recurring SLA monitoring loop

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                   EVENT BUS                          │
  ├─────────┬──────────┬───────────┬───────────────────┤
  │  Pipeline│  Quality │  Schema   │  SLA              │
  │  Agent   │  Agent   │  Agent    │  Agent            │
  └─────────┴──────────┴───────────┴───────────────────┘
"""
