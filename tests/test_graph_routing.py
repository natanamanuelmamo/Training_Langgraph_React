"""Router tests (``task02.md`` §8).

The routers are pure ``state -> node name`` functions with no side effects, so every branch of
the graph's control flow is covered here without building a graph at all. That is the payoff
for keeping them in their own module.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph", reason="Part 2 requires the optional 'graph' extra")

from react_langgraph.routing import route_after_decide, route_after_reason
from react_langgraph.state import TriageState, initial_state

pytestmark = pytest.mark.graph


def _state(**overrides: object) -> TriageState:
    state = initial_state("LINE", "t")
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ---------------------------------------------------------------------------
# route_after_reason
# ---------------------------------------------------------------------------


def test_routes_to_act_when_a_tool_was_chosen() -> None:
    assert route_after_reason(_state(iteration=1, next_action="classify_severity"), 6) == "act"


def test_routes_to_decide_when_final() -> None:
    assert route_after_reason(_state(iteration=3, is_final=True), 6) == "decide"


def test_routes_to_halt_at_the_iteration_cap() -> None:
    assert route_after_reason(_state(iteration=6, next_action="classify_severity"), 6) == "halt"


def test_routes_to_halt_past_the_iteration_cap() -> None:
    assert route_after_reason(_state(iteration=9, next_action="x"), 6) == "halt"


def test_final_wins_over_the_iteration_cap() -> None:
    """Concluding on the last allowed turn must still yield a complete decision."""
    assert route_after_reason(_state(iteration=6, is_final=True), 6) == "decide"


def test_routes_back_to_reason_when_no_action_was_parsed() -> None:
    """Malformed output is a reasoning retry, not a no-op action."""
    assert route_after_reason(_state(iteration=2, next_action=""), 6) == "reason"
    assert route_after_reason(_state(iteration=2), 6) == "reason"


def test_the_retry_self_loop_is_still_bounded_by_the_cap() -> None:
    """Rule R10: the self-loop must not be able to spin forever."""
    assert route_after_reason(_state(iteration=6, next_action=""), 6) == "halt"


# ---------------------------------------------------------------------------
# route_after_decide — the human-in-the-loop gate
# ---------------------------------------------------------------------------


def test_routes_to_the_sensitive_node_only_when_approval_is_required() -> None:
    assert route_after_decide(_state(requires_approval=True)) == "page_on_call"


def test_routes_to_notify_when_no_approval_is_required() -> None:
    assert route_after_decide(_state(requires_approval=False)) == "notify"


def test_rejection_diverts_away_from_the_sensitive_node() -> None:
    """The single line that turns the checkpoint into a decision (``task02.md`` §5.5)."""
    state = _state(requires_approval=True, approval="rejected")
    assert route_after_decide(state) == "notify"


def test_approval_keeps_the_sensitive_route() -> None:
    state = _state(requires_approval=True, approval="approved")
    assert route_after_decide(state) == "page_on_call"


def test_rejection_on_a_non_sensitive_run_is_harmless() -> None:
    state = _state(requires_approval=False, approval="rejected")
    assert route_after_decide(state) == "notify"


# ---------------------------------------------------------------------------
# routers are pure
# ---------------------------------------------------------------------------


def test_routers_do_not_mutate_state() -> None:
    state = _state(iteration=2, next_action="classify_severity", requires_approval=True)
    before = dict(state)
    route_after_reason(state, 6)
    route_after_decide(state)
    assert dict(state) == before
