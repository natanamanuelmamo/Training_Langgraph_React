"""Conditional-edge functions (``task02.md`` §3).

Part 2 only. These are the graph's control flow made explicit: plain, pure,
separately-testable functions that take state and return a node name. No side effects, no I/O,
no LangGraph import — which is why ``tests/test_graph_routing.py`` can cover every branch
without building a graph at all.

:func:`route_after_reason` is where Part 1's ``while`` condition ended up.
:func:`route_after_decide` is the human-in-the-loop gate.
"""

from __future__ import annotations

from typing import Literal

from react_langgraph.state import TriageState

AfterReason = Literal["act", "decide", "halt", "reason"]
AfterDecide = Literal["page_on_call", "notify"]


def route_after_reason(state: TriageState, max_iterations: int) -> AfterReason:
    """Decide what happens after a reasoning turn.

    Order matters. ``is_final`` is checked before the iteration cap so a model that concludes
    on its last allowed turn still yields a complete decision instead of being halted one step
    from the finish line.

    Args:
        state: The current state.
        max_iterations: The cap from settings (Rule R10).

    Returns:
        ``decide`` when the model has concluded, ``halt`` at the cap, ``reason`` to retry after
        unparseable output, and ``act`` otherwise.
    """
    if state["is_final"]:
        return "decide"
    if state["iteration"] >= max_iterations:
        return "halt"
    if not state.get("next_action"):
        # The model produced no usable Action — malformed output, already recorded as an
        # Observation. Reason again rather than routing through a no-op `act`; the iteration
        # cap above still bounds this.
        return "reason"
    return "act"


def route_after_decide(state: TriageState) -> AfterDecide:
    """Decide whether the sensitive node runs.

    The rejection check comes **first**, and that ordering is the whole point: it is what turns
    the checkpoint from a prompt into a decision. When a human has rejected the page, this
    router sends the run down the non-paging path, so ``page_on_call`` never executes at all
    (``task02.md`` §5.5).

    Args:
        state: The current state, after ``decide`` has run.

    Returns:
        ``page_on_call`` only for an approved sensitive action; ``notify`` otherwise.
    """
    if state.get("approval") == "rejected":
        return "notify"
    if state["requires_approval"]:
        return "page_on_call"
    return "notify"
