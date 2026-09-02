"""Part 1 loop tests (``task01.md`` §7).

These are the tests that prove the ReAct pattern actually works: the loop terminates, the
scratchpad accumulates and feeds forward, every failure mode becomes an Observation rather than
a crash, and the iteration cap produces a partial result instead of hanging.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from tests.conftest import CERT_LINE, INFO_LINE, OOM_LINE, POOL_LINE, WARN_LINE

from react_from_scratch.agent import run_react_agent
from react_from_scratch.scratchpad import Scratchpad
from triage_core.domain.models import Action, Severity, StopReason, TriageRun
from triage_core.infra.cache import ResultCache
from triage_core.infra.tracing import Tracer
from triage_core.llm.client import FakeLLM, ScriptedLLM
from triage_core.tools.base import Tool

Runner = Callable[..., TriageRun]


def _final(action: str = "file_ticket", severity: str = "critical") -> str:
    return (
        "Thought: I have everything I need.\n"
        f'Final Answer: {{"action": "{action}", "severity": "{severity}", "confidence": 0.86,'
        ' "justification": "Because the policy said so.", "matched_incidents": ["INC-0001"]}'
    )


def _classify(line: str) -> str:
    import json

    return (
        "Thought: classify first\nAction: classify_severity\n"
        f"Action Input: {json.dumps({'log_line': line})}"
    )


# ---------------------------------------------------------------------------
# termination
# ---------------------------------------------------------------------------


def test_loop_terminates_on_a_final_answer(run_part1: Runner) -> None:
    run = run_part1(POOL_LINE, responses=[_final()])
    assert run.decision.complete
    assert run.decision.stop_reason is StopReason.FINAL_ANSWER
    assert run.decision.action is Action.FILE_TICKET
    assert run.iterations == 1
    assert run.llm_calls == 1
    assert run.tool_calls == 0


def test_loop_stops_at_the_iteration_cap_with_a_partial_result(run_part1: Runner) -> None:
    """Rule R10: never spin, never raise — return a partial result and say why."""
    never_finishes = [_classify(POOL_LINE)] * 6
    run = run_part1(POOL_LINE, responses=never_finishes, max_iterations=3)

    assert not run.decision.complete
    assert run.decision.stop_reason is StopReason.ITERATION_LIMIT
    assert run.iterations == 3
    assert run.llm_calls == 3
    assert "3-iteration limit" in run.decision.justification


def test_partial_result_keeps_the_severity_it_managed_to_observe(run_part1: Runner) -> None:
    run = run_part1(POOL_LINE, responses=[_classify(POOL_LINE)] * 3, max_iterations=2)
    assert not run.decision.complete
    assert run.decision.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# error recovery — every failure becomes an Observation, and the loop continues
# ---------------------------------------------------------------------------


def test_unknown_tool_becomes_an_observation_and_the_loop_continues(
    run_part1: Runner,
) -> None:
    script = [
        "Thought: try something\nAction: summon_wizard\nAction Input: {}",
        _final(),
    ]
    run = run_part1(POOL_LINE, responses=script)

    observations = [s.text for s in run.trace if s.kind == "observation"]
    assert any("Unknown tool 'summon_wizard'" in text for text in observations)
    assert any("classify_severity" in text for text in observations)
    assert run.decision.complete
    assert run.iterations == 2


def test_tool_input_error_becomes_an_observation_and_the_loop_continues(
    run_part1: Runner,
) -> None:
    script = [
        'Thought: bad args\nAction: lookup_incidents\nAction Input: {"signature": "s"}',
        _final(),
    ]
    run = run_part1(POOL_LINE, responses=script)

    observations = [s.text for s in run.trace if s.kind == "observation"]
    assert any("Tool error:" in text and "severity" in text for text in observations)
    assert run.decision.complete


def test_malformed_llm_output_becomes_an_observation_and_the_loop_continues(
    run_part1: Runner,
) -> None:
    script = ["I refuse to follow any format whatsoever.", _final()]
    run = run_part1(POOL_LINE, responses=script)

    observations = [s.text for s in run.trace if s.kind == "observation"]
    assert any("Output format error" in text for text in observations)
    assert run.decision.complete
    assert run.iterations == 2


def test_invalid_action_input_json_becomes_an_observation(run_part1: Runner) -> None:
    script = [
        "Thought: oops\nAction: classify_severity\nAction Input: {not json}",
        _final(),
    ]
    run = run_part1(POOL_LINE, responses=script)
    observations = [s.text for s in run.trace if s.kind == "observation"]
    assert any("was not valid JSON" in text for text in observations)
    assert run.decision.complete


def test_unexpected_argument_names_become_an_observation(run_part1: Runner) -> None:
    script = [
        'Thought: wrong keys\nAction: classify_severity\nAction Input: {"line": "x"}',
        _final(),
    ]
    run = run_part1(POOL_LINE, responses=script)
    observations = [s.text for s in run.trace if s.kind == "observation"]
    assert any("Tool error:" in text for text in observations)
    assert run.decision.complete


def test_a_genuine_tool_bug_is_not_disguised_as_a_bad_argument(
    registry: dict[str, Tool], fresh_cache: ResultCache
) -> None:
    """Only the declared failure classes become Observations.

    A tool blowing up with something else is a bug in the tool, and hiding it behind an
    Observation would leave the agent politely looping around a defect.
    """

    class _BrokenTool:
        name: ClassVar[str] = "classify_severity"
        description: ClassVar[str] = "d"
        input_schema: ClassVar[dict[str, str]] = {}

        def run(self, **kwargs: object) -> object:
            raise ZeroDivisionError("a real bug, not a bad argument")

    broken = dict(registry) | {"classify_severity": _BrokenTool()}
    llm = ScriptedLLM([_classify(POOL_LINE), _final()])

    with pytest.raises(ZeroDivisionError):
        run_react_agent(
            POOL_LINE,
            llm=llm,
            tools=broken,  # type: ignore[arg-type]
            cache=fresh_cache,
            tracer=Tracer(),
        )


# ---------------------------------------------------------------------------
# the scratchpad is what makes it an agent
# ---------------------------------------------------------------------------


def test_the_scratchpad_grows_and_feeds_the_next_reason_step(
    registry: dict[str, Tool], fresh_cache: ResultCache
) -> None:
    """Iteration n's prompt must contain iteration n-1's Observation."""
    llm = ScriptedLLM([_classify(POOL_LINE), _final()])
    run_react_agent(
        POOL_LINE, llm=llm, tools=registry, cache=fresh_cache, tracer=Tracer(), max_iterations=6
    )

    first_prompt = llm.calls[0][1]
    second_prompt = llm.calls[1][1]

    assert "Observation:" not in first_prompt
    assert "severity=critical" in second_prompt
    assert len(second_prompt) > len(first_prompt)
    assert POOL_LINE in first_prompt


def test_scratchpad_render_matches_the_react_grammar() -> None:
    pad = Scratchpad("LINE")
    pad.begin_iteration(1)
    pad.add_thought("thinking")
    pad.add_action("classify_severity", {"log_line": "LINE"})
    pad.add_observation("severity=info signature=s")

    rendered = pad.render()
    assert "Log line: LINE" in rendered
    assert "Thought: thinking" in rendered
    assert "Action: classify_severity" in rendered
    assert '"log_line": "LINE"' in rendered
    assert "Observation: severity=info signature=s" in rendered


def test_system_prompt_is_stable_across_iterations(
    registry: dict[str, Tool], fresh_cache: ResultCache
) -> None:
    """Only the transcript grows — the system prompt does not."""
    llm = ScriptedLLM([_classify(POOL_LINE), _final()])
    run_react_agent(
        POOL_LINE, llm=llm, tools=registry, cache=fresh_cache, tracer=Tracer(), max_iterations=6
    )
    assert llm.calls[0][0] == llm.calls[1][0]


# ---------------------------------------------------------------------------
# every policy branch, end to end, through the real loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "severity", "action", "rule"),
    [
        (POOL_LINE, Severity.CRITICAL, Action.FILE_TICKET, "critical_known_fix"),
        (OOM_LINE, Severity.CRITICAL, Action.PAGE_ON_CALL, "critical_regression"),
        (CERT_LINE, Severity.CRITICAL, Action.PAGE_ON_CALL, "critical_novel"),
        (WARN_LINE, Severity.WARNING, Action.FILE_TICKET, "warning_ticket"),
        (INFO_LINE, Severity.INFO, Action.IGNORE, "info_ignore"),
    ],
)
def test_every_decision_branch_end_to_end(
    run_part1: Runner, line: str, severity: Severity, action: Action, rule: str
) -> None:
    """The full loop, driven by the offline FakeLLM, on one line per branch."""
    run = run_part1(line, llm=FakeLLM())

    assert run.decision.complete
    assert run.decision.severity is severity
    assert run.decision.action is action
    assert run.iterations == 4
    assert run.tool_calls == 3
    assert any(rule in step.text for step in run.trace if step.kind == "observation")


def test_the_full_loop_produces_the_expected_trace_shape(run_part1: Runner) -> None:
    run = run_part1(POOL_LINE, llm=FakeLLM())
    kinds = [step.kind for step in run.trace]

    assert kinds == [
        "thought",
        "action",
        "observation",
        "thought",
        "action",
        "observation",
        "thought",
        "action",
        "observation",
        "thought",
        "final",
    ]


# ---------------------------------------------------------------------------
# defaults and wiring
# ---------------------------------------------------------------------------


def test_loop_runs_without_an_explicit_cache_or_tracer(registry: dict[str, Tool]) -> None:
    run = run_react_agent(POOL_LINE, llm=FakeLLM(), tools=registry)
    assert run.decision.complete
    assert run.cache_hits == 0


def test_a_bad_final_answer_payload_does_not_crash_the_loop(
    registry: dict[str, Tool], fresh_cache: ResultCache
) -> None:
    """The parser validates shape; conversion failures still recover gracefully."""

    class _Weird:
        """Emits a Final Answer that parses but cannot convert, then a good one."""

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return (
                    'Thought: done\nFinal Answer: {"action": "ignore", "severity": "info",'
                    ' "confidence": 0.5, "justification": "j", "matched_incidents": [1, 2]}'
                )
            return _final()

    run = run_react_agent(
        POOL_LINE,
        llm=_Weird(),  # type: ignore[arg-type]
        tools=registry,
        cache=fresh_cache,
        tracer=Tracer(),
    )
    assert run.decision.complete


def test_run_is_json_serialisable(run_part1: Runner) -> None:
    import json

    run = run_part1(POOL_LINE, llm=FakeLLM())
    payload: dict[str, Any] = run.to_dict()
    assert json.loads(json.dumps(payload))["decision"]["action"] == "file_ticket"
