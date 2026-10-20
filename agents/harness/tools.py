"""Tool definitions — the only surface an agent can act through.

A Tool pairs a JSON-schema contract (what the model or policy sees) with a
Python handler (what actually runs). Every tool declares a Permission tier so
the harness can decide, before anything executes, whether the call is allowed.

    READ         → observe only (metrics, profiles, schema inspection)
    WRITE        → changes pipeline state (run a stage, quarantine a batch)
    DESTRUCTIVE  → overwrites published data (publish to Silver/Gold paths)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        return {"read": 0, "write": 1, "destructive": 2}[self.value]


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass
class Tool:
    """A single capability exposed to the agent.

    Attributes:
        name: Unique tool identifier (snake_case, model-facing).
        description: What the tool does and when to use it (model-facing).
        handler: Callable receiving validated keyword arguments.
        permission: Tier used by guardrails to allow or block the call.
        input_schema: JSON Schema for the arguments (object type).
        idempotent: Safe to retry without side effects compounding.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    permission: Permission = Permission.READ
    input_schema: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    idempotent: bool = True

    def validate(self, args: dict) -> Optional[str]:
        """Return an error message if args violate the schema, else None.

        Deliberately small: checks required keys, unknown keys, and primitive
        types. Enough to reject malformed model output without a jsonschema
        dependency.
        """
        properties = self.input_schema.get("properties", {})
        for key in self.input_schema.get("required", []):
            if key not in args:
                return f"missing required argument '{key}'"
        for key, value in args.items():
            if key not in properties:
                return f"unknown argument '{key}'"
            expected = properties[key].get("type")
            if expected and not isinstance(value, _JSON_TYPES[expected]):
                return f"argument '{key}' must be of type {expected}"
            # bool is a subclass of int; reject it where a number is expected
            if expected in ("integer", "number") and isinstance(value, bool):
                return f"argument '{key}' must be of type {expected}"
            allowed = properties[key].get("enum")
            if allowed and value not in allowed:
                return f"argument '{key}' must be one of {allowed}"
        return None

    def spec(self) -> dict:
        """Model-facing tool spec (Anthropic Messages API `tools` format)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Holds the bounded tool set a harness exposes."""

    def __init__(self, tools: Optional[list[Tool]] = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, max_permission: Permission = Permission.DESTRUCTIVE) -> list[dict]:
        """Specs for tools at or below a permission tier."""
        return [
            t.spec() for t in self._tools.values()
            if t.permission.rank <= max_permission.rank
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
