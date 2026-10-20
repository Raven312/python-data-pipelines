"""Unit tests for ClaudePolicy with a scripted fake Anthropic client."""

from types import SimpleNamespace

from agents.harness import AgentHarness, GuardrailPolicy, Tool, ToolRegistry, approve_all
from agents.loop import AgentLoop, LoopStatus
from agents.loop.claude_policy import FINISH_TOOL, ClaudePolicy


def tool_use(name, tool_input=None, id_="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input or {}, id=id_)


def text(value):
    return SimpleNamespace(type="text", text=value)


class FakeClient:
    """Returns scripted responses and records every request."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return SimpleNamespace(content=self._turns.pop(0))


def harness():
    reg = ToolRegistry([
        Tool("run_bronze", "Ingest", lambda: {"rows": 10}),
        Tool("run_gold", "Aggregate", lambda: (_ for _ in ()).throw(RuntimeError("503"))),
    ])
    return AgentHarness(reg, GuardrailPolicy.operator(approver=approve_all))


def test_tools_include_finish_and_harness_specs():
    h = harness()
    policy = ClaudePolicy(h.tool_specs(), client=FakeClient([]))
    assert [t["name"] for t in policy.tools] == ["run_bronze", "run_gold", "finish"]
    assert FINISH_TOOL in policy.tools


def test_drives_loop_and_feeds_back_tool_results():
    client = FakeClient([
        [text("Ingest first."), tool_use("run_bronze", id_="tu_1")],
        [tool_use("run_gold", id_="tu_2")],
        [tool_use("finish", {"status": "escalated", "reason": "gold keeps failing"}, id_="tu_3")],
    ])
    h = harness()
    result = AgentLoop(h, ClaudePolicy(h.tool_specs(), client=client)).run("load suppliers")

    assert result.status == LoopStatus.ESCALATED
    assert [s.action.tool for s in result.steps] == ["run_bronze", "run_gold"]
    assert result.steps[0].action.rationale == "Ingest first."

    first, second, third = client.requests
    assert first["messages"][0] == {"role": "user", "content": "Goal: load suppliers"}
    assert first["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}

    ok_result = second["messages"][-1]["content"][0]
    assert ok_result["tool_use_id"] == "tu_1" and ok_result["is_error"] is False
    assert '"rows": 10' in ok_result["content"]

    err_result = third["messages"][-1]["content"][0]
    assert err_result["tool_use_id"] == "tu_2" and err_result["is_error"] is True
    assert "503" in err_result["content"]


def test_no_tool_call_fails_safely():
    client = FakeClient([[text("I am not sure.")]])
    h = harness()
    result = AgentLoop(h, ClaudePolicy(h.tool_specs(), client=client)).run("load")
    assert result.status == LoopStatus.FAILED


def test_model_cannot_bypass_guardrails():
    reg = ToolRegistry([Tool("drop_table", "dangerous", lambda: None)])
    h = AgentHarness(reg, GuardrailPolicy(denylist={"drop_table"}))
    client = FakeClient([
        [tool_use("drop_table", id_="tu_1")],
        [tool_use("finish", {"status": "blocked", "reason": "refused"}, id_="tu_2")],
    ])
    result = AgentLoop(h, ClaudePolicy(h.tool_specs(), client=client)).run("cleanup")
    assert result.steps[0].result.blocked
    assert result.status == LoopStatus.BLOCKED


def test_hallucinated_tool_is_blocked():
    client = FakeClient([
        [tool_use("run_platinum", id_="tu_1")],
        [tool_use("finish", {"status": "succeeded", "reason": "done"}, id_="tu_2")],
    ])
    h = harness()
    loop = AgentLoop(h, ClaudePolicy(h.tool_specs(), client=client),
                     verify=lambda steps: None if any(s.result.ok for s in steps) else "nothing ran")
    result = loop.run("load")
    assert "unknown tool" in result.steps[0].result.error
    assert result.status == LoopStatus.FAILED
