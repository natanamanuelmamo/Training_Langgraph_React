"""The graph's shared state (``task02.md`` §2).

Part 2 only. ``TriageState`` is what flows between nodes and what the checkpointer persists.

Two rules govern every node that touches it:

* Nodes return **partial** updates — ``{"iteration": 3}`` — never the whole dict.
* ``messages`` is the only reduced field: ``operator.add`` appends. Every other field is
  last-write-wins, which is why counters are written as absolute values read off the cache
  rather than as ``state[...] + 1``. One source of truth, no drift.

This module imports nothing from LangGraph. The reducer is built from ``operator.add`` and
``typing.Annotated``, both stdlib — keeping ``triage_core``'s neighbours framework-light and
making the dependency on LangGraph visible in exactly one place (``builder.py``).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from triage_core.domain.models import Incident, Severity, TraceStep, TriageDecision

Approval = Literal["approved", "rejected"]


class TriageState(TypedDict):
    """Everything one triage run knows, threaded through the graph."""

    # ---- input ----
    log_line: str
    thread_id: str

    # ---- accumulated reasoning ----
    #: Appended to by every node. Carries both the transcript steps replayed into the next
    #: reason call (thought / action / observation) and the node rows the CLI prints.
    messages: Annotated[list[TraceStep], operator.add]
    iteration: int

    # ---- tool outputs ----
    severity: NotRequired[Severity]
    signature: NotRequired[str]
    incidents: NotRequired[list[Incident]]

    # ---- control ----
    next_action: NotRequired[str]
    next_action_input: NotRequired[dict[str, Any]]
    is_final: bool
    #: The raw parsed ``Final Answer``, carried from ``reason`` to ``decide`` so that the one
    #: node responsible for building the decision is the one that validates it.
    final_payload: NotRequired[dict[str, Any]]

    # ---- decision + approval ----
    decision: NotRequired[TriageDecision]
    requires_approval: bool
    approval: NotRequired[Approval]
    approval_note: NotRequired[str]

    # ---- observability ----
    cache_hits: int
    cache_misses: int
    llm_calls: int
    tool_calls: int


def initial_state(log_line: str, thread_id: str) -> TriageState:
    """Build the starting state for a run.

    Every non-``NotRequired`` key must be present before the first node runs.

    Args:
        log_line: The raw log line to triage.
        thread_id: Checkpointer key for this run.

    Returns:
        A fully populated initial state.
    """
    return TriageState(
        log_line=log_line,
        thread_id=thread_id,
        messages=[],
        iteration=0,
        is_final=False,
        requires_approval=False,
        cache_hits=0,
        cache_misses=0,
        llm_calls=0,
        tool_calls=0,
    )


def transcript_steps(state: TriageState) -> list[TraceStep]:
    """The subset of ``messages`` that forms the model-facing transcript.

    Node rows exist for the printed trace only and must not leak into the prompt, or Part 1
    and Part 2 would send different text for the same run and the parity test would be
    meaningless.

    Args:
        state: The current state.

    Returns:
        Only the thought / action / observation steps, in order.
    """
    return [step for step in state["messages"] if step.kind in {"thought", "action", "observation"}]


def observations(state: TriageState) -> list[str]:
    """Every Observation text recorded so far, oldest first."""
    return [step.text for step in state["messages"] if step.kind == "observation"]
