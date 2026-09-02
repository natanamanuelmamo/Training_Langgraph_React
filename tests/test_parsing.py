"""Table-driven tests for ``parse_react_step`` (``task01.md`` §7).

The parser is pure, so these are the cheapest tests in the suite and the ones that protect the
loop from every malformed-output case.
"""

from __future__ import annotations

import pytest

from triage_core.domain.errors import ParseError
from triage_core.llm.parsing import parse_react_step

GOOD_ACTION = """\
Thought: I need the severity first.
Action: classify_severity
Action Input: {"log_line": "ERROR payments-api pool exhausted"}
"""

GOOD_FINAL = """\
Thought: I have everything I need.
Final Answer: {"action": "file_ticket", "severity": "critical", "confidence": 0.86,
               "justification": "Known failure with a prior fix.",
               "matched_incidents": ["INC-2291"]}
"""


def test_parses_a_tool_call() -> None:
    step = parse_react_step(GOOD_ACTION)
    assert not step.is_final
    assert step.thought == "I need the severity first."
    assert step.action == "classify_severity"
    assert step.action_input == {"log_line": "ERROR payments-api pool exhausted"}


def test_parses_a_final_answer() -> None:
    step = parse_react_step(GOOD_FINAL)
    assert step.is_final
    assert step.final_answer is not None
    assert step.final_answer["action"] == "file_ticket"
    assert step.final_answer["matched_incidents"] == ["INC-2291"]


def test_tolerates_code_fences() -> None:
    step = parse_react_step(f"```\n{GOOD_ACTION}```")
    assert step.action == "classify_severity"


def test_tolerates_leading_and_trailing_prose() -> None:
    raw = f"Sure, here is my next step.\n\n{GOOD_ACTION}\nLet me know if that helps."
    step = parse_react_step(raw)
    assert step.action == "classify_severity"


def test_parses_multi_line_action_input() -> None:
    raw = """\
Thought: Look it up.
Action: lookup_incidents
Action Input: {
    "signature": "db.pool.exhausted:orders-primary",
    "severity": "critical"
}
"""
    step = parse_react_step(raw)
    assert step.action_input == {
        "signature": "db.pool.exhausted:orders-primary",
        "severity": "critical",
    }


def test_final_answer_wins_when_both_are_present() -> None:
    """The model has concluded; erroring here would burn an iteration.

    See ``docs/task_1_implementation.md`` §11.
    """
    raw = GOOD_ACTION + "\n" + GOOD_FINAL
    step = parse_react_step(raw)
    assert step.is_final


def test_handles_braces_inside_strings() -> None:
    raw = 'Thought: t\nAction: classify_severity\nAction Input: {"log_line": "a } b { c"}'
    step = parse_react_step(raw)
    assert step.action_input == {"log_line": "a } b { c"}


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        ("", "Empty response"),
        ("   \n  ", "Empty response"),
        ("Action: classify_severity\nAction Input: {}", "No 'Thought:' line"),
        ("Thought: hmm, let me think about it.", "No 'Action:' or 'Final Answer:'"),
        ("Thought: t\nAction: classify_severity", "no 'Action Input:' line"),
        ("Thought: t\nAction:\nAction Input: {}", "was empty"),
        ("Thought: t\nAction: x\nAction Input: not json at all", "must be a JSON object"),
        ('Thought: t\nAction: x\nAction Input: {"a": 1,}', "was not valid JSON"),
        ('Thought: t\nAction: x\nAction Input: {"a": 1', "unterminated JSON object"),
        ('Thought: t\nAction: x\nAction Input: ["a"]', "must be a JSON object"),
    ],
)
def test_malformed_output_raises_a_helpful_parse_error(raw: str, needle: str) -> None:
    """The message is handed back to the model verbatim, so it must say what was wrong."""
    with pytest.raises(ParseError) as excinfo:
        parse_react_step(raw)
    assert needle in str(excinfo.value)


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ('{"action": "file_ticket"}', "missing required key"),
        (
            '{"action": "nuke_it", "severity": "critical", "confidence": 0.5,'
            ' "justification": "j", "matched_incidents": []}',
            "'action' must be one of",
        ),
        (
            '{"action": "ignore", "severity": "spicy", "confidence": 0.5,'
            ' "justification": "j", "matched_incidents": []}',
            "'severity' must be one of",
        ),
        (
            '{"action": "ignore", "severity": "info", "confidence": 4.2,'
            ' "justification": "j", "matched_incidents": []}',
            "must be between 0 and 1",
        ),
        (
            '{"action": "ignore", "severity": "info", "confidence": "high",'
            ' "justification": "j", "matched_incidents": []}',
            "must be a number",
        ),
        (
            '{"action": "ignore", "severity": "info", "confidence": 0.5,'
            ' "justification": "j", "matched_incidents": "INC-1"}',
            "must be a list",
        ),
    ],
)
def test_final_answer_validation(payload: str, needle: str) -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_react_step(f"Thought: done\nFinal Answer: {payload}")
    assert needle in str(excinfo.value)


def test_parser_is_pure() -> None:
    """Same input, same output — no hidden state between calls."""
    first = parse_react_step(GOOD_ACTION)
    second = parse_react_step(GOOD_ACTION)
    assert first == second
