"""Human-in-the-loop: inspect the pause, approve, reject, resume (``task02.md`` §5).

Part 2 only. ``page_on_call`` is the sensitive action, and the graph is compiled with
``interrupt_before=["page_on_call"]`` so it pauses *before* running it.

Approve and reject are deliberately **asymmetric**, and that asymmetry is load-bearing::

    approve  →  update_state(...)                    then invoke(None)
    reject   →  update_state(..., as_node="decide")  then invoke(None)

**Reject uses ``as_node="decide"``** so the write is attributed to the ``decide`` node, which
makes LangGraph re-evaluate ``route_after_decide`` from that checkpoint. The router sees
``approval == "rejected"`` and returns ``"notify"``, so ``page_on_call`` never runs at all.
Without ``as_node`` the update lands as a plain patch, the pending task for ``page_on_call``
survives, and resuming pages the engineer anyway — a checkpoint that merely *delays* the same
result, which ``task02.md`` §5.5 explicitly rejects.

**Approve must not use ``as_node``**, because re-running the edge would re-enter
``page_on_call`` as a fresh pending task and ``interrupt_before`` would pause on it again — a
pause loop. A plain patch records the approval for the audit trail and lets the resume proceed.

Both behaviours are verified against the installed LangGraph in ``tests/test_graph_hitl.py``.
"""

from __future__ import annotations

import textwrap
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from react_langgraph.builder import SENSITIVE_NODE
from react_langgraph.state import TriageState
from triage_core.domain.models import TriageDecision

App = CompiledStateGraph[TriageState, None, TriageState, TriageState]
RunConfig = RunnableConfig


def make_config(thread_id: str, max_iterations: int) -> RunConfig:
    """Build the run config for a thread.

    Args:
        thread_id: Checkpointer key; the same id resumes the same run.
        max_iterations: Our own cap, used to size LangGraph's recursion backstop.

    Returns:
        A config carrying both the thread id and the recursion limit.
    """
    from react_langgraph.builder import recursion_limit

    return RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=recursion_limit(max_iterations),
    )


def is_paused(app: App, config: RunConfig) -> bool:
    """Whether the run is paused before the sensitive node.

    Args:
        app: The compiled graph.
        config: The run config naming the thread.

    Returns:
        True when ``state.next`` names the sensitive node.
    """
    return SENSITIVE_NODE in app.get_state(config).next


def pending_state(app: App, config: RunConfig) -> TriageState:
    """The state as of the pause.

    Args:
        app: The compiled graph.
        config: The run config naming the thread.

    Returns:
        The persisted state values.
    """
    # LangGraph types its persisted values as a loose mapping; we know the schema because we
    # declared it when compiling the graph.
    return cast("TriageState", app.get_state(config).values)


def pending_summary(state: TriageState) -> str:
    """Render what a human needs in order to say yes or no.

    Everything a reviewer must weigh at 3am: the proposed action, the evidence behind it, and
    the incidents it matched.

    Args:
        state: The paused state.

    Returns:
        A printable summary block.
    """
    decision: TriageDecision | None = state.get("decision")
    if decision is None:
        return "  (no decision recorded)"

    incidents = ", ".join(decision.matched_incidents) or "none"
    lines = [
        f"  Proposed action : {decision.action}",
        f"  Severity        : {decision.severity}   confidence {decision.confidence:.2f}",
        f"  Signature       : {state.get('signature', 'unknown')}",
        f"  Matched incident: {incidents}",
        "  Why             :",
    ]
    lines += [f"      {line}" for line in textwrap.wrap(decision.justification, width=84)]
    return "\n".join(lines)


def approve(app: App, config: RunConfig) -> TriageState:
    """Approve the page and resume, executing the sensitive node.

    Deliberately a *plain* state patch — see this module's docstring for why ``as_node`` would
    cause a pause loop here.

    Args:
        app: The compiled graph.
        config: The run config naming the paused thread.

    Returns:
        The final state.
    """
    app.update_state(config, {"approval": "approved"})
    return cast("TriageState", app.invoke(None, config=config))


def reject(app: App, config: RunConfig, note: str = "") -> TriageState:
    """Reject the page and resume down the non-paging path.

    Attributed to ``decide`` so the routing edge is re-evaluated and the run is diverted —
    ``page_on_call`` never executes. See this module's docstring.

    Args:
        app: The compiled graph.
        config: The run config naming the paused thread.
        note: The reviewer's reason, recorded in the rewritten justification.

    Returns:
        The final state, whose decision differs from the approved outcome.
    """
    app.update_state(
        config,
        {"approval": "rejected", "approval_note": note},
        as_node="decide",
    )
    return cast("TriageState", app.invoke(None, config=config))


def prompt_for_approval(summary: str) -> bool:
    """Ask a human at the terminal. Used only by the CLI.

    Anything other than an explicit yes is a refusal — the safe default for an action that
    wakes someone up.

    Args:
        summary: The rendered pending summary to show first.

    Returns:
        True only when the reviewer explicitly approves.
    """
    print(summary)
    try:
        answer = input("   Approve page to on-call? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}
