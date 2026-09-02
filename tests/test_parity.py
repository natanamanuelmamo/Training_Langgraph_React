"""Part 1 and Part 2 must agree (``task02.md`` §8).

This is the test that enforces Rule R3. The two parts orchestrate differently — a hand-written
``while`` loop versus a ``StateGraph`` — but they share every tool, prompt, parser and policy.
Given the same scripted model responses they must therefore reach the *same* decision.

If someone forks the escalation policy into ``react_langgraph/``, or lets the two parts drift
apart in how they render the transcript, this file goes red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="Part 2 requires the optional 'graph' extra")

from tests.conftest import (
    CERT_LINE,
    INFO_LINE,
    OOM_LINE,
    POOL_LINE,
    TODAY,
    WARN_LINE,
)
from tests.graph_helpers import make_harness

from react_from_scratch.agent import run_react_agent
from react_langgraph.hitl import approve, is_paused
from triage_core.domain.models import TriageDecision
from triage_core.domain.prompts import SCRATCHPAD_HEADER
from triage_core.infra.cache import ResultCache
from triage_core.infra.tracing import Tracer, render_transcript
from triage_core.llm.client import ScriptedLLM
from triage_core.tools.registry import build_registry, describe_tools

pytestmark = pytest.mark.graph


def _script(log_line: str, severity: str, signature: str, action: str) -> list[str]:
    """A fixed four-step ReAct script, identical for both parts."""
    import json

    return [
        "Thought: classify it first\nAction: classify_severity\n"
        f"Action Input: {json.dumps({'log_line': log_line})}",
        "Thought: has this happened before?\nAction: lookup_incidents\n"
        f"Action Input: {json.dumps({'signature': signature, 'severity': severity})}",
        "Thought: apply the policy\nAction: recommend_action\n"
        f"Action Input: {json.dumps({'severity': severity, 'incidents': []})}",
        "Thought: done\n"
        f'Final Answer: {{"action": "{action}", "severity": "{severity}",'
        ' "confidence": 0.86, "justification": "Agreed by both implementations.",'
        ' "matched_incidents": ["INC-0001"]}',
    ]


def _run_part1(incidents_file: Path, log_line: str, script: list[str]) -> TriageDecision:
    tools = build_registry(
        llm=ScriptedLLM(script), incidents_path=incidents_file, lookback_days=90, today=TODAY
    )
    run = run_react_agent(
        log_line,
        llm=ScriptedLLM(script),
        tools=tools,
        cache=ResultCache(enabled=True),
        tracer=Tracer(),
        max_iterations=6,
    )
    return run.decision


def _run_part2(incidents_file: Path, log_line: str, script: list[str]) -> TriageDecision:
    harness = make_harness(incidents_file, responses=script, thread_id="parity")
    state = harness.start(log_line)
    if is_paused(harness.app, harness.config):
        state = approve(harness.app, harness.config)
    decision: TriageDecision = state["decision"]
    return decision


@pytest.mark.parametrize(
    ("log_line", "severity", "signature", "action"),
    [
        (POOL_LINE, "critical", "db.pool.exhausted:orders-primary", "file_ticket"),
        (OOM_LINE, "critical", "oom.killed:recommendation-worker", "page_on_call"),
        (CERT_LINE, "critical", "tls.cert.expired:checkout-gateway", "page_on_call"),
        (WARN_LINE, "warning", "net.timeout:search-api", "file_ticket"),
        (INFO_LINE, "info", "cache.warm:catalog-api", "ignore"),
    ],
)
def test_both_parts_reach_the_same_decision(
    incidents_file: Path, log_line: str, severity: str, signature: str, action: str
) -> None:
    """Same script in, same ``TriageDecision`` out — however the loop is orchestrated."""
    script = _script(log_line, severity, signature, action)

    part1 = _run_part1(incidents_file, log_line, script)
    part2 = _run_part2(incidents_file, log_line, script)

    assert part1 == part2, f"Part 1 said {part1}, Part 2 said {part2}"


def test_both_parts_send_the_model_identical_prompts(incidents_file: Path) -> None:
    """The transcript is shared code, so the two parts must produce byte-identical text.

    Without this the parity above could pass by luck — the scripted LLM ignores its input, so
    a divergence in prompt rendering would go unnoticed until a real model was used.
    """
    script = _script(POOL_LINE, "critical", "db.pool.exhausted:orders-primary", "file_ticket")

    llm1 = ScriptedLLM(script)
    tools = build_registry(llm=llm1, incidents_path=incidents_file, lookback_days=90, today=TODAY)
    run_react_agent(
        POOL_LINE,
        llm=llm1,
        tools=tools,
        cache=ResultCache(enabled=True),
        tracer=Tracer(),
        max_iterations=6,
    )

    harness = make_harness(incidents_file, responses=script, thread_id="prompt-parity")
    harness.start(POOL_LINE)
    llm2 = harness.deps.llm
    assert isinstance(llm2, ScriptedLLM)

    assert len(llm1.calls) == len(llm2.calls) == 4
    for turn, (call1, call2) in enumerate(zip(llm1.calls, llm2.calls, strict=True)):
        assert call1[0] == call2[0], f"system prompts diverged on turn {turn + 1}"
        assert call1[1] == call2[1], f"transcripts diverged on turn {turn + 1}"


def test_both_parts_use_the_same_tool_registry_type(incidents_file: Path) -> None:
    """Rule R3: one tool implementation, imported by both — never duplicated."""
    harness = make_harness(incidents_file)
    tools = build_registry(
        llm=ScriptedLLM([]), incidents_path=incidents_file, lookback_days=90, today=TODAY
    )

    assert set(harness.deps.tools) == set(tools)
    for name in tools:
        assert type(harness.deps.tools[name]) is type(tools[name])
        assert harness.deps.tools[name].__class__.__module__.startswith("triage_core.tools")


def test_the_transcript_renderer_is_shared(incidents_file: Path) -> None:
    """Both parts build their prompt from the same function and the same header."""
    from react_from_scratch.scratchpad import Scratchpad

    pad = Scratchpad("LINE")
    pad.begin_iteration(1)
    pad.add_thought("t")
    expected = SCRATCHPAD_HEADER + "\n" + render_transcript("LINE", pad.steps)
    assert pad.render() == expected


def test_both_parts_describe_the_tools_identically(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    assert describe_tools(harness.deps.tools) == describe_tools(harness.deps.tools)
    assert "recommend_action" in describe_tools(harness.deps.tools)
