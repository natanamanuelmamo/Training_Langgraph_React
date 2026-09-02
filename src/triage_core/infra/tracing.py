"""Trace accumulation and rendering.

Serves both parts (Rule R8: the trace is a deliverable, not debug output).

Two different renderings of the same :class:`TraceStep` list live here:

* :func:`render_transcript` — the *prompt* form, fed back into the next reason step. Part 1's
  ``Scratchpad`` and Part 2's ``reason`` node both call it, so both parts send the model
  byte-identical prompts. That is what makes ``test_parity.py`` meaningful rather than
  coincidental.
* :func:`render_trace` and :func:`render_summary` — the *display* form a human reads.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Sequence

from triage_core.domain.models import TraceStep, TriageRun

_RULE_WIDTH = 58
_BODY_WIDTH = 92
_LABEL_WIDTH = 8


class Tracer:
    """Collects :class:`TraceStep` objects as a run proceeds.

    Part 1 accumulates here. Part 2 accumulates in the graph state's ``messages`` list instead,
    then renders the result through the same functions.
    """

    def __init__(self) -> None:
        """Start an empty trace."""
        self._steps: list[TraceStep] = []

    def add(self, step: TraceStep) -> None:
        """Append one step.

        Args:
            step: The step to record.
        """
        self._steps.append(step)

    @property
    def steps(self) -> tuple[TraceStep, ...]:
        """Every step recorded so far, in order."""
        return tuple(self._steps)


def render_transcript(log_line: str, steps: Iterable[TraceStep]) -> str:
    """Render the transcript that is fed back into the next reason step.

    This is the agent's memory: a stateless model call behaves like an agent only because this
    string grows every iteration.

    Args:
        log_line: The line under triage.
        steps: The thought / action / observation steps so far.

    Returns:
        The transcript, in the same grammar the model is asked to produce.
    """
    lines = [f"Log line: {log_line}", ""]
    for step in steps:
        if step.kind == "thought":
            lines.append(f"Thought: {step.text}")
        elif step.kind == "action":
            lines.append(f"Action: {step.label}")
            lines.append(f"Action Input: {step.text}")
        elif step.kind == "observation":
            lines.append(f"Observation: {step.text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _wrap(label: str, body: str, *, indent: int = 4) -> list[str]:
    """Render one ``label : body`` display line, wrapping and indenting continuations."""
    prefix = " " * indent + f"{label:<{_LABEL_WIDTH}}: "
    hanging = " " * len(prefix)
    out: list[str] = []
    for position, paragraph in enumerate(body.splitlines() or [""]):
        wrapped = textwrap.wrap(paragraph, width=_BODY_WIDTH) or [""]
        for offset, piece in enumerate(wrapped):
            first = position == 0 and offset == 0
            out.append((prefix if first else hanging) + piece)
    return out


def render_trace(steps: Sequence[TraceStep], *, title: str, log_line: str) -> str:
    """Render the human-readable reasoning trace.

    Args:
        steps: The steps recorded during the run.
        title: Banner text, e.g. ``ReAct (raw Python)``.
        log_line: The line under triage, echoed under the banner.

    Returns:
        The full trace, ready to print.
    """
    out: list[str] = [
        f"── {title} " + "─" * max(0, _RULE_WIDTH - len(title) - 4),
        f"Input: {log_line}",
        "",
    ]

    current = 0
    for step in steps:
        if step.kind == "thought":
            if current and step.index != current:
                out.append("")
            current = step.index
            out.append(f"[{step.index}] {'Thought':<{_LABEL_WIDTH}}: {step.text}")
        elif step.kind == "action":
            out.extend(_wrap("Action", f"{step.label} {step.text}"))
        elif step.kind == "observation":
            label = "Observ." if step.cache is None else f"Observ. [{step.cache.upper()}]"
            out.extend(_wrap(label, step.text))
        elif step.kind == "final":
            out.extend(_wrap("Final", step.text))
        elif step.kind == "node":
            out.append(f"▸ {step.label:<12} [{step.index}] {step.text}")
        elif step.kind == "notice":
            out.extend(_wrap("Note", step.text))
    return "\n".join(out)


def render_node_trace(steps: Sequence[TraceStep], *, title: str, thread_id: str) -> str:
    """Render Part 2's compact node-transition trace (``task02.md`` §6).

    A different shape from :func:`render_trace` because the two specs ask for different things:
    Part 1 shows the Thought/Action/Observation cycle, Part 2 shows the graph walking its nodes
    and looping back. Only ``node`` and ``notice`` steps are printed; the thought/action/
    observation steps in the same list are the model-facing transcript.

    Args:
        steps: The steps accumulated in the graph state's ``messages``.
        title: Banner text.
        thread_id: The checkpointer key, echoed under the banner.

    Returns:
        The trace, ready to print.
    """
    out = [
        f"── {title} " + "─" * max(0, _RULE_WIDTH - len(title) - 4),
        f"thread_id: {thread_id}",
        "",
    ]
    for step in steps:
        if step.kind == "node":
            out.append(f"▸ {step.label:<13} [{step.index}]  {step.text}")
        elif step.kind == "notice":
            out.append(f"  {step.text}")
    return "\n".join(out)


def render_summary(run: TriageRun) -> str:
    """Render the result block and the counters line (Rule R8).

    Args:
        run: The completed run.

    Returns:
        The summary block, ready to print.
    """
    decision = run.decision
    incidents = ", ".join(decision.matched_incidents) or "none"
    out = [
        "── Result " + "─" * (_RULE_WIDTH - 10),
        f"Action        : {decision.action}",
        f"Severity      : {decision.severity}",
        f"Confidence    : {decision.confidence:.2f}",
    ]
    justification = textwrap.wrap(decision.justification, width=_BODY_WIDTH) or [""]
    out.append(f"Justification : {justification[0]}")
    out.extend(" " * 16 + piece for piece in justification[1:])
    out.append(f"Incidents     : {incidents}")
    if not decision.complete:
        out.append(f"Incomplete    : stopped because {decision.stop_reason}")
    out.append(
        f"Iterations    : {run.iterations}    Tool calls: {run.tool_calls}    "
        f"LLM calls: {run.llm_calls}    "
        f"Cache: {run.cache_hits} hit / {run.cache_misses} miss"
    )
    return "\n".join(out)
