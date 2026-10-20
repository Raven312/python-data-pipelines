"""Audit trail — an append-only record of every action the agent attempted.

Each record captures who (session), when (step, timestamp), what (tool, args),
the guardrail verdict, and the outcome with latency. Records can be mirrored
to a JSON Lines file so a run can be replayed or loaded into a Delta table
for incident review.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    session_id: str
    step: int
    tool_name: str
    args: dict
    status: str  # "ok" | "error" | "blocked"
    permission: Optional[str] = None
    latency_ms: float = 0.0
    output_summary: Any = None
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AuditTrail:
    """In-memory audit log with optional JSON Lines persistence."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.records: list[ActionRecord] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: ActionRecord) -> None:
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), default=str) + "\n")

    def by_status(self, status: str) -> list[ActionRecord]:
        return [r for r in self.records if r.status == status]

    def summary(self) -> dict:
        tools: dict[str, int] = {}
        for r in self.records:
            tools[r.tool_name] = tools.get(r.tool_name, 0) + 1
        return {
            "total_actions": len(self.records),
            "ok": len(self.by_status("ok")),
            "errors": len(self.by_status("error")),
            "blocked": len(self.by_status("blocked")),
            "calls_by_tool": tools,
            "total_latency_ms": round(sum(r.latency_ms for r in self.records), 1),
        }
