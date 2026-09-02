"""Parse the model's text into a :class:`ReActStep` (``task01.md`` §4).

Serves both parts: Part 1's loop and Part 2's ``reason`` node both call
:func:`parse_react_step`. This is the "reasoning parsed by our own parser" the assignment asks
for — no framework does it for us.

Pure: no I/O, no logging, no recovery. It raises :class:`ParseError` with a message precise
enough to be handed straight back to the model as an Observation, and the *caller* decides what
to do with it. That is how malformed output becomes a correction rather than a crash (R10).
"""

from __future__ import annotations

import json
import re
from typing import Any

from triage_core.domain.errors import ParseError
from triage_core.domain.models import ReActStep
from triage_core.domain.prompts import FINAL_ANSWER_KEYS, VALID_ACTIONS, VALID_SEVERITIES

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n?")
_THOUGHT = re.compile(r"^[ \t]*Thought[ \t]*:[ \t]*(?P<body>.*)$", re.IGNORECASE | re.MULTILINE)
_ACTION = re.compile(r"^[ \t]*Action[ \t]*:[ \t]*(?P<body>.*)$", re.IGNORECASE | re.MULTILINE)
_ACTION_INPUT = re.compile(r"^[ \t]*Action[ _]?Input[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)
_FINAL_ANSWER = re.compile(r"^[ \t]*Final[ _]?Answer[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences, which models add unprompted."""
    return _FENCE.sub("", text)


def _extract_json_object(text: str, *, label: str) -> dict[str, Any]:
    """Read one JSON object starting at the first ``{``, matching braces to find its end.

    A brace scan rather than a line read, so a pretty-printed multi-line object parses.

    Args:
        text: The text following the label, e.g. everything after ``Action Input:``.
        label: The label being parsed, used in error messages.

    Returns:
        The decoded object.

    Raises:
        ParseError: No object was found, it was unbalanced, it was not valid JSON, or the
            top-level value was not an object. The message names the offending text so the
            model can correct itself.
    """
    start = text.find("{")
    if start == -1:
        snippet = text.strip()[:160]
        raise ParseError(f"{label} must be a JSON object starting with '{{'. Received: {snippet!r}")

    depth = 0
    in_string = False
    escaped = False
    end = -1
    for position, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = position + 1
                break

    if end == -1:
        raise ParseError(
            f"{label} has an unterminated JSON object — a closing '}}' is missing. "
            f"Received: {text[start : start + 160]!r}"
        )

    blob = text[start:end]
    try:
        decoded = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"{label} was not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}. "
            f"Received: {blob[:160]!r}"
        ) from exc

    if not isinstance(decoded, dict):
        raise ParseError(
            f"{label} must be a JSON object, got {type(decoded).__name__}. Received: {blob[:160]!r}"
        )
    return decoded


def _validate_final_answer(payload: dict[str, Any]) -> None:
    """Check the Final Answer carries every required key with a usable value.

    Raises:
        ParseError: A key is missing or a value is outside its allowed set.
    """
    if missing := [key for key in FINAL_ANSWER_KEYS if key not in payload]:
        raise ParseError(
            f"Final Answer is missing required key(s): {missing}. "
            f"It must contain exactly: {list(FINAL_ANSWER_KEYS)}"
        )

    action = str(payload["action"]).lower()
    if action not in VALID_ACTIONS:
        raise ParseError(
            f"Final Answer 'action' must be one of {list(VALID_ACTIONS)}, "
            f"got {payload['action']!r}."
        )

    severity = str(payload["severity"]).lower()
    if severity not in VALID_SEVERITIES:
        raise ParseError(
            f"Final Answer 'severity' must be one of {list(VALID_SEVERITIES)}, "
            f"got {payload['severity']!r}."
        )

    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise ParseError(
            f"Final Answer 'confidence' must be a number between 0 and 1, "
            f"got {payload['confidence']!r}."
        ) from exc
    if not 0.0 <= confidence <= 1.0:
        raise ParseError(f"Final Answer 'confidence' must be between 0 and 1, got {confidence}.")

    if not isinstance(payload["matched_incidents"], list):
        raise ParseError(
            "Final Answer 'matched_incidents' must be a list of incident id strings, got "
            f"{type(payload['matched_incidents']).__name__}."
        )


def parse_react_step(raw: str) -> ReActStep:
    """Parse one model response into a Thought plus an Action or a Final Answer.

    Tolerant of surrounding prose, code fences and whitespace. When a response carries both an
    ``Action`` and a ``Final Answer``, the Final Answer wins — the model has concluded, and
    erroring would burn an iteration (see ``docs/task_1_implementation.md`` §11).

    Args:
        raw: The model's raw text response.

    Returns:
        The parsed step.

    Raises:
        ParseError: The response does not match the required grammar. The message is written
            to be fed back to the model verbatim as an Observation.
    """
    if not raw or not raw.strip():
        raise ParseError(
            "Empty response. Expected 'Thought:' followed by either 'Action:' and "
            "'Action Input:', or 'Final Answer:'."
        )

    text = _strip_fences(raw)

    thought_match = _THOUGHT.search(text)
    if thought_match is None:
        raise ParseError(
            "No 'Thought:' line found. Every response must begin with 'Thought: <reasoning>'. "
            f"Received: {raw.strip()[:160]!r}"
        )
    thought = thought_match.group("body").strip()

    if final_match := _FINAL_ANSWER.search(text):
        payload = _extract_json_object(text[final_match.end() :], label="Final Answer")
        _validate_final_answer(payload)
        return ReActStep(thought=thought, final_answer=payload)

    action_match = _ACTION.search(text)
    if action_match is None:
        raise ParseError(
            "No 'Action:' or 'Final Answer:' line found. After the Thought you must either "
            "call a tool ('Action:' plus 'Action Input:') or finish ('Final Answer:')."
        )

    action = action_match.group("body").strip().strip("`\"'")
    if not action:
        raise ParseError("'Action:' was empty. It must name exactly one tool.")

    input_match = _ACTION_INPUT.search(text, action_match.end())
    if input_match is None:
        raise ParseError(
            f"Found 'Action: {action}' but no 'Action Input:' line. Every Action needs an "
            "'Action Input:' line containing a JSON object."
        )

    action_input = _extract_json_object(text[input_match.end() :], label="Action Input")
    return ReActStep(thought=thought, action=action, action_input=action_input)
