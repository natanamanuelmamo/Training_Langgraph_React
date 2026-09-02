"""Graph behaviour tests: the loop-back, the cap, the cache, and the HITL checkpoint.

These cover the behaviours ``task02.md`` §8 grades. The most important is the reject path: a
checkpoint that only delays the same result is not a checkpoint, so we assert both that the
sensitive node never executed (via a spy counter, because inspecting the final state alone
would not catch a node that ran and was then overwritten) and that the outcome differs.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="Part 2 requires the optional 'graph' extra")

from tests.conftest import CERT_LINE, INFO_LINE, OOM_LINE, POOL_LINE, WARN_LINE
from tests.graph_helpers import make_harness

from react_langgraph.builder import (
    SENSITIVE_NODE,
    build_app,
    recursion_limit,
    render_mermaid,
)
from react_langgraph.hitl import (
    approve,
    is_paused,
    pending_state,
    pending_summary,
    reject,
)
from triage_core.domain.models import Action, Severity, StopReason
from triage_core.infra.cache import ResultCache

pytestmark = pytest.mark.graph


# ---------------------------------------------------------------------------
# the ReAct loop, expressed as a graph
# ---------------------------------------------------------------------------


def test_the_graph_loops_back_from_act_to_reason(incidents_file: Path) -> None:
    """The edge the assignment explicitly asks for. At least two loop-backs must occur."""
    harness = make_harness(incidents_file)
    harness.start(POOL_LINE)

    labels = harness.node_labels()
    loop_backs = sum(
        1 for first, second in pairwise(labels) if first == "act" and second == "reason"
    )
    assert loop_backs >= 2, f"expected the act -> reason edge to fire twice, got {labels}"


def test_a_full_run_reaches_a_decision(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    state = harness.start(POOL_LINE)

    assert state["decision"].action is Action.FILE_TICKET
    assert state["decision"].complete
    assert state["iteration"] == 4
    assert state["tool_calls"] == 3
    assert state["llm_calls"] == 4


def test_structured_tool_output_lands_in_state(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    state = harness.start(POOL_LINE)

    assert state["severity"] is Severity.CRITICAL
    assert state["signature"] == "db.pool.exhausted:orders-primary"
    assert [i.incident_id for i in state["incidents"]] == ["INC-0002", "INC-0001"]


def test_the_iteration_cap_routes_to_halt(incidents_file: Path) -> None:
    """Rule R10: terminate cleanly with a partial result rather than spinning."""
    never_finishes = [
        f'Thought: again\nAction: classify_severity\nAction Input: {{"log_line": "{POOL_LINE}"}}'
    ] * 12
    harness = make_harness(incidents_file, responses=never_finishes, max_iterations=4)
    state = harness.start(POOL_LINE)

    assert "halt" in harness.node_labels()
    assert SENSITIVE_NODE not in harness.node_labels()
    assert not state["decision"].complete
    assert state["decision"].stop_reason is StopReason.ITERATION_LIMIT
    assert state["iteration"] == 4


def test_unparseable_output_loops_back_through_reason(incidents_file: Path) -> None:
    """A parse failure is a reasoning retry, recorded as an Observation."""
    good_final = (
        'Thought: done\nFinal Answer: {"action": "file_ticket", "severity": "warning",'
        ' "confidence": 0.8, "justification": "j", "matched_incidents": []}'
    )
    harness = make_harness(incidents_file, responses=["no format at all", good_final])
    state = harness.start(WARN_LINE)

    texts = [s.text for s in state["messages"] if s.kind == "observation"]
    assert any("Output format error" in t for t in texts)
    assert state["decision"].complete
    assert harness.node_labels()[:2] == ["reason", "reason"]


def test_an_unknown_tool_becomes_an_observation(incidents_file: Path) -> None:
    good_final = (
        'Thought: done\nFinal Answer: {"action": "ignore", "severity": "info",'
        ' "confidence": 0.9, "justification": "j", "matched_incidents": []}'
    )
    script = ["Thought: try\nAction: summon_wizard\nAction Input: {}", good_final]
    harness = make_harness(incidents_file, responses=script)
    state = harness.start(INFO_LINE)

    texts = [s.text for s in state["messages"] if s.kind == "observation"]
    assert any("Unknown tool 'summon_wizard'" in t for t in texts)
    assert state["decision"].complete


# ---------------------------------------------------------------------------
# the interrupt
# ---------------------------------------------------------------------------


def test_the_graph_pauses_before_the_sensitive_node(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)

    assert harness.app.get_state(harness.config).next == (SENSITIVE_NODE,)
    assert is_paused(harness.app, harness.config)
    assert harness.pages_sent == 0

    state = pending_state(harness.app, harness.config)
    assert state["requires_approval"] is True
    assert state["decision"].action is Action.PAGE_ON_CALL


def test_the_pending_summary_shows_a_human_what_they_are_approving(
    incidents_file: Path,
) -> None:
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)
    summary = pending_summary(pending_state(harness.app, harness.config))

    assert "page_on_call" in summary
    assert "critical" in summary
    assert "tls.cert.expired:checkout-gateway" in summary


def test_approving_resumes_and_executes_the_sensitive_node(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)

    state = approve(harness.app, harness.config)

    assert harness.pages_sent == 1
    assert SENSITIVE_NODE in harness.node_labels()
    assert state["decision"].action is Action.PAGE_ON_CALL
    assert state["approval"] == "approved"
    assert harness.app.get_state(harness.config).next == ()


def test_rejecting_never_executes_the_sensitive_node(incidents_file: Path) -> None:
    """The spy counter is the real assertion here — see this module's docstring."""
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)

    reject(harness.app, harness.config, "cert renewal already in flight")

    assert harness.pages_sent == 0
    assert SENSITIVE_NODE not in harness.node_labels()
    assert "notify" in harness.node_labels()


def test_rejecting_changes_the_outcome(incidents_file: Path) -> None:
    """A checkpoint that only delays the same result is not a checkpoint."""
    approved = make_harness(incidents_file, thread_id="approve")
    approved.start(CERT_LINE)
    approved_state = approve(approved.app, approved.config)

    rejected = make_harness(incidents_file, thread_id="reject")
    rejected.start(CERT_LINE)
    rejected_state = reject(rejected.app, rejected.config, "renewal in flight")

    assert approved_state["decision"].action is Action.PAGE_ON_CALL
    assert rejected_state["decision"].action is Action.FILE_TICKET
    assert approved_state["decision"] != rejected_state["decision"]
    assert rejected_state["decision"].stop_reason is StopReason.HUMAN_REJECTED
    assert "renewal in flight" in rejected_state["decision"].justification


def test_the_reviewers_note_is_recorded(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)
    state = reject(harness.app, harness.config, "known maintenance window")

    assert state["approval"] == "rejected"
    assert state["approval_note"] == "known maintenance window"


def test_rejecting_without_a_note_still_works(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)
    state = reject(harness.app, harness.config, "")

    assert state["decision"].action is Action.FILE_TICKET
    assert "no reason given" in state["decision"].justification


@pytest.mark.parametrize("line", [WARN_LINE, INFO_LINE])
def test_non_sensitive_paths_never_interrupt(incidents_file: Path, line: str) -> None:
    harness = make_harness(incidents_file)
    state = harness.start(line)

    assert harness.app.get_state(harness.config).next == ()
    assert not is_paused(harness.app, harness.config)
    assert harness.pages_sent == 0
    assert state["requires_approval"] is False
    assert state["decision"].complete


def test_a_recurring_critical_failure_also_pauses(incidents_file: Path) -> None:
    """The regression branch pages too, so it must be gated as well."""
    harness = make_harness(incidents_file)
    harness.start(OOM_LINE)

    assert is_paused(harness.app, harness.config)
    assert pending_state(harness.app, harness.config)["decision"].action is Action.PAGE_ON_CALL


# ---------------------------------------------------------------------------
# checkpointer
# ---------------------------------------------------------------------------


def test_threads_are_isolated_from_each_other(incidents_file: Path) -> None:
    harness = make_harness(incidents_file, thread_id="thread-a")
    harness.start(WARN_LINE)

    from react_langgraph.hitl import make_config

    other = make_config("thread-b", 6)
    assert harness.app.get_state(other).values == {}


def test_the_pause_is_persisted_state_not_a_blocked_call(incidents_file: Path) -> None:
    """Inspecting the thread repeatedly must keep showing the same pause."""
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)

    first = harness.app.get_state(harness.config)
    second = harness.app.get_state(harness.config)

    assert first.next == second.next == (SENSITIVE_NODE,)
    assert harness.pages_sent == 0
    assert second.values["decision"].action is Action.PAGE_ON_CALL


def test_domain_objects_survive_the_checkpoint_round_trip(incidents_file: Path) -> None:
    """The checkpointer must restore our dataclasses, not lists of raw fields."""
    harness = make_harness(incidents_file)
    harness.start(CERT_LINE)
    values = harness.app.get_state(harness.config).values

    assert isinstance(values["decision"].action, Action)
    assert isinstance(values["severity"], Severity)
    assert isinstance(values["decision"].matched_incidents, tuple)


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------


def test_the_second_identical_lookup_is_a_cache_hit(incidents_file: Path) -> None:
    """One cache shared across two runs — the demonstration ``task02.md`` §4 asks for."""
    cache = ResultCache(enabled=True)
    first = make_harness(incidents_file, cache=cache, thread_id="run-1")
    first.start(POOL_LINE)
    assert (cache.stats.hits, cache.stats.misses) == (0, 1)

    second = make_harness(incidents_file, cache=cache, thread_id="run-2")
    state = second.start(POOL_LINE)

    assert (cache.stats.hits, cache.stats.misses) == (1, 1)
    assert state["cache_hits"] == 1
    verdicts = [s.cache for s in state["messages"] if s.cache is not None]
    assert verdicts == ["hit"]


def test_no_cache_produces_two_misses(incidents_file: Path) -> None:
    cache = ResultCache(enabled=False)
    first = make_harness(incidents_file, cache=cache, cache_enabled=False, thread_id="a")
    first.start(POOL_LINE)
    second = make_harness(incidents_file, cache=cache, cache_enabled=False, thread_id="b")
    state = second.start(POOL_LINE)

    assert (cache.stats.hits, cache.stats.misses) == (0, 2)
    assert state["cache_misses"] == 2


def test_cache_counters_reach_the_state(incidents_file: Path) -> None:
    harness = make_harness(incidents_file)
    state = harness.start(POOL_LINE)
    assert state["cache_misses"] == 1
    assert state["cache_hits"] == 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_recursion_limit_leaves_room_for_our_own_cap() -> None:
    """LangGraph's backstop must never fire before the ``halt`` node does."""
    assert recursion_limit(6) == 18
    assert recursion_limit(4) == 14


def test_the_mermaid_diagram_shows_the_loop_back_and_the_interrupt(
    incidents_file: Path,
) -> None:
    harness = make_harness(incidents_file)
    diagram = render_mermaid(harness.app)

    assert "act --> reason" in diagram
    assert "page_on_call" in diagram
    assert "interrupt" in diagram


def test_build_app_accepts_an_injected_checkpointer(incidents_file: Path) -> None:
    from react_langgraph.builder import build_checkpointer

    harness = make_harness(incidents_file)
    app = build_app(harness.deps, checkpointer=build_checkpointer())
    assert app is not harness.app
