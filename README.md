# Agentic Medallion ETL Pipeline (PySpark + Databricks)

An enterprise-style data pipeline that an AI agent can **operate safely**.

- A **PySpark Medallion pipeline** (Bronze → Silver → Gold) with data quality gates, metrics and Databricks-native modules (Delta Live Tables, Delta Lake operations, Unity Catalog, Auto Loader, Workflows).
- An **agent harness** that wraps the pipeline in a small set of tools. Each tool has a permission tier and a call budget, and every call is validated, hooked and audited.
- An **agent loop** that drives the pipeline stage by stage, retries transient failures, quarantines bad batches, escalates what it can't fix, and only counts a run as done once the publish step has really succeeded. A recurring **monitoring loop** watches SLAs and starts these runs when data goes stale.

The decision-maker can be swapped: a deterministic rule-based policy by default, or Claude via tool use. Either way the harness enforces the same limits.

---

## Architecture

```
                         ┌──────────────────────────────────────┐
                         │          MONITORING LOOP             │  every N minutes
                         │  SLA checks → remediate? → escalate? │  (scheduler / cron)
                         └──────────────────┬───────────────────┘
                                            │ starts a run when freshness /
                                            ▼ completeness is breached
┌───────────────────┐   Action    ┌─────────────────────────────────────────────┐
│      POLICY       │ ──────────> │                AGENT HARNESS                │
│ RuleBasedPolicy   │             │ guardrails → pre-hooks → tool → post-hooks  │
│ or ClaudePolicy   │ <────────── │                  → audit                    │
└───────────────────┘ ToolResult  └──────────────────────┬──────────────────────┘
          ▲                                              │ bounded tools only
          │            AGENT LOOP                        ▼
          │  observe → decide → act → (repeat)   ┌──────────────────────────────┐
          └───────────────────────────────────── │       PIPELINE TOOLKIT       │
             stop: finish · steps · time ·       │ inspect_source_schema        │
                   consecutive failures          │ run_bronze  → run_bronze_gate│
                                                 │ run_silver  → run_silver_gate│
                                                 │ profile_silver_quality       │
                                                 │ run_gold    → publish_outputs│
                                                 │ quarantine_batch · escalate  │
                                                 └──────────────┬───────────────┘
                                                                ▼
                         ┌────────┐  DQ gate  ┌────────┐  DQ gate  ┌────────┐
                         │ BRONZE │ ────────> │ SILVER │ ────────> │  GOLD  │
                         └────────┘           └────────┘           └────────┘
```

---

## Harness + Loop design

### Why a harness?

Giving an agent (especially an LLM) direct access to Spark, storage or credentials is how pipelines get corrupted. The harness is the **only** way an agent can act, and it answers four questions before anything runs:

| Question | Mechanism | Where |
|---|---|---|
| *Can the agent do this at all?* | Tool registry. Unknown or made-up tools are refused | `agents/harness/tools.py` |
| *Is it allowed to?* | Permission tiers `READ < WRITE < DESTRUCTIVE`, a per-agent ceiling, a denylist | `agents/harness/guardrails.py` |
| *Has it done too much?* | Per-tool and total call budgets | `agents/harness/guardrails.py` |
| *Is the request well-formed?* | JSON-schema argument validation (rejects malformed model output) | `Tool.validate` |
| *Does a human need to sign off?* | Approval gate on gated tiers (publishing to Gold is `DESTRUCTIVE`) | `GuardrailPolicy.approver` |

Around each allowed call the harness runs **hooks** (`pre_tool` can veto, for example during a maintenance window; `post_tool` and `on_error` handle metrics and alerting) and writes an **audit record** to memory and optionally to JSON Lines (`logs/agent_audit.jsonl`). Each record holds the session, step, tool, arguments, permission, status, latency and an output summary.

`AgentHarness.invoke()` **never raises**. Success, a handler exception and a guardrail refusal all come back as a `ToolResult`, so the loop can see the failure and react to it.

### Tools exposed to the agent

| Tool | Tier | What it does |
|---|---|---|
| `inspect_source_schema` | READ | Schema-drift check on the incoming batch. Additive columns are *evolved* (rescued into `_rescued_data`, Auto Loader style). Dropped columns are *rejected* |
| `run_bronze` / `run_bronze_gate` | WRITE | Ingest with audit columns, then run the Bronze quality gate |
| `run_silver` / `run_silver_gate` | WRITE | Dedupe, cleanse, join, conform, then run the Silver quality gate |
| `profile_silver_quality` | READ | Z-score anomaly check against the learned baseline (`DataQualityAgent`) |
| `run_gold` | WRITE | Build the Gold aggregates |
| `publish_outputs` | **DESTRUCTIVE** | Overwrite the published Silver/Gold outputs. Needs approval |
| `quarantine_batch` | WRITE | Park the offending batch plus `reason.json` under `quarantine/<stage>/<run_id>` |
| `get_pipeline_state` | READ | Completed stages, row counts, quarantines, output paths |
| `escalate` | READ | Hand off to a human with a root-cause summary (changes no data) |

Stage tools check their own preconditions (Silver needs a passed Bronze gate, publish needs Gold), so a confused planner gets a clear error instead of a bad write.

### The agent loop

`AgentLoop` (`agents/loop/agent_loop.py`) splits the work three ways:

- **Loop:** control flow and stop conditions (`max_steps`, wall-clock budget, consecutive failures).
- **Policy:** judgment, meaning which action comes next given the observation.
- **Harness:** safety, meaning whether that action may run.

```
while not stopped:
    observation = goal + full history of (action, result)
    decision    = policy.decide(observation)      # Action(tool, args, wait) | Finish(status)
    result      = harness.invoke(decision.tool)    # guarded, hooked, audited
```

Before accepting a `SUCCEEDED` finish, the loop runs a `verify` check against the actual history: a run only succeeds if `publish_outputs` really succeeded. The loop trusts results, not the policy's claim, which matters once the policy is an LLM.

Terminal statuses: `SUCCEEDED`, `ESCALATED` (deliberate hand-off), `BLOCKED` (a guardrail or a human said no), `BUDGET_EXHAUSTED`, `FAILED`.

### Policies

**`RuleBasedPolicy`** (`agents/loop/policy.py`) is deterministic, replayable and offline. It classifies failures using the existing `PipelineOrchestrationAgent`:

| Failure signal | Classified as | Action |
|---|---|---|
| `503`, throttling, rate limit | transient | retry with exponential backoff (2s, 4s, 8s), max 3 |
| timeout | timeout | retry with backoff |
| OOM, executor or node lost | infrastructure | restart the stage (max 2) |
| quality gate failure | data quality | `quarantine_batch` → `escalate` |
| breaking schema drift | schema | `escalate` before any write |
| anything else | unknown | `escalate` |

**`ClaudePolicy`** (`agents/loop/claude_policy.py`) is a drop-in LLM planner. Claude receives `harness.tool_specs()` (already filtered by the permission ceiling) plus a `finish` tool and picks one tool per turn (`tool_choice: any`, no parallel calls). Tool results, including errors and guardrail refusals, are fed back as `tool_result` blocks. If the model invents a tool, tries a denylisted one or skips approval, the harness blocks it the same way it would block the rule-based policy.

### The monitoring loop

`MonitoringLoop` (`agents/loop/monitor.py`) is the recurring outer loop. On each tick it:

1. **Observes** the SLA checks (freshness, latency, availability, completeness) through `SLAMonitorAgent`.
2. **Decides** what to do. Pass or warn means nothing. A breach on a metric that a re-run can fix (freshness, completeness) means **remediate**. Any other breach means **alert**.
3. **Acts** by running the agent loop and recording the outcome back into the SLA history, so the next tick sees the recovery.
4. **Guards** against endless self-healing. After *N* failed remediations, or as soon as the agent escalates, it alerts once and pauses until a human calls `acknowledge()`.

In production a scheduler triggers `tick()`, for example a Databricks Workflows job on a cron schedule or an Airflow sensor. `run(cycles=…)` with an injected `sleep` is there for demos and tests.

### Incident replay

Every agent run produces a transcript and an `incident_report()` covering the outcome, the step where the incident was detected, the failed tool, the error, the recovery actions and the duration. The integration tests replay each scenario against real Spark:

| Injected fault | Detected at | Agent response | Outcome |
|---|---|---|---|
| `503 throttled` on Silver | step 4 | retry with 2s backoff | ✅ succeeded |
| Empty supplier batch | step 3 (Bronze gate) | quarantine → escalate | 🟡 escalated, nothing published |
| New upstream column `s_comment` | pre-flight | rescue into `_rescued_data`, continue | ✅ succeeded |
| Dropped column `s_phone` | pre-flight | escalate before ingest | 🟡 escalated, nothing written |
| Publish approval denied | step 8 | stop | ⛔ blocked, Gold untouched |
| Unknown executor error on Gold | step 7 | escalate | 🟡 escalated |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs Java 17+ for PySpark

# Classic batch run
python main.py

# Agentic run: the agent loop drives each stage through the harness
python main.py --mode agentic                       # prompts before publishing
python main.py --mode agentic --auto-approve

# Replay an incident
python main.py --mode agentic --inject-fault transient --auto-approve
python main.py --mode agentic --inject-fault empty_batch
python main.py --mode agentic --inject-fault additive_column --auto-approve
python main.py --mode agentic --inject-fault dropped_column

# Claude as the planner (pip install anthropic; export ANTHROPIC_API_KEY=...)
python main.py --mode agentic --policy claude --auto-approve

# Recurring monitoring loop with agentic remediation
python main.py --mode monitor --cycles 3 --interval 5 --auto-approve
```

### Programmatic use

```python
from agents.harness import approve_all
from agents.loop.operator import run_agentic_pipeline

result, toolkit = run_agentic_pipeline(
    spark, "data/output", supplier_data, nation_data,
    policy="rules",                       # or "claude", or any object with .decide()
    approver=approve_all,                 # human gate for DESTRUCTIVE tools
    audit_path="logs/agent_audit.jsonl",
)
print("\n".join(result.transcript()))
print(result.incident_report())
```

---

## Project structure

```
├── main.py                         # CLI: batch | agentic | monitor
├── dashboard.py                    # Terminal executive dashboard
├── config/settings.py              # Multi-environment configuration
├── pipeline/                       # PySpark Medallion pipeline
│   ├── bronze/ingestion.py         #   raw ingest + audit columns
│   ├── silver/transformations.py   #   dedupe, cleanse, join, SCD2
│   ├── gold/aggregations.py        #   business aggregates
│   ├── quality/checks.py           #   composable data quality gates
│   ├── orchestrator/pipeline_runner.py
│   └── utils/                      #   metrics, SparkSession factory
├── agents/
│   ├── harness/                    # ★ Agent harness
│   │   ├── tools.py                #   Tool, Permission, ToolRegistry
│   │   ├── guardrails.py           #   GuardrailPolicy, approval gate
│   │   ├── hooks.py                #   pre/post/error lifecycle hooks
│   │   ├── audit.py                #   AuditTrail (JSONL)
│   │   ├── harness.py              #   AgentHarness.invoke()
│   │   └── pipeline_toolkit.py     #   pipeline stages as tools, fault injection
│   ├── loop/                       # ★ Agent loop
│   │   ├── agent_loop.py           #   AgentLoop, Observation, LoopResult
│   │   ├── policy.py               #   RuleBasedPolicy
│   │   ├── claude_policy.py        #   ClaudePolicy (LLM planner)
│   │   ├── monitor.py              #   MonitoringLoop (recurring SLA loop)
│   │   └── operator.py             #   wiring: toolkit + harness + policy + loop
│   ├── orchestrator/               # failure triage agent (used by the policy)
│   ├── quality/                    # statistical anomaly agent
│   ├── schema/                     # schema drift agent
│   └── sla/                        # SLA monitor agent
├── databricks/                     # DLT tables, Delta ops, Unity Catalog,
│                                   # Auto Loader, SQL analytics, Workflows
├── src/                            # Legacy pandas pipeline (kept for reference)
└── tests/
    ├── unit/                       # incl. harness, loop, policy, monitor
    └── integration/                # incl. agentic incident replays on Spark
```

## Testing

```bash
pytest                                   # full suite (starts a local Spark session)
pytest tests/unit/test_harness.py tests/unit/test_agent_loop.py \
       tests/unit/test_claude_policy.py tests/unit/test_monitor.py   # no Spark session
pytest tests/integration/test_agentic_pipeline.py                    # incident replays
```

The harness, loop, policy and monitor tests use fake tools and a fake Anthropic client, so they run in well under a second without starting Spark or calling the network.

## License

MIT
