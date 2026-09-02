"""Graph wiring and compilation (``task02.md`` §3, §5).

Part 2 only. **This is the only module in the project that imports LangGraph.** Part 1 imports
``triage_core``, so a framework import anywhere else would fail ``tests/test_isolation.py`` and
invalidate Part 1 (Rule R1).

The shape::

              START → reason ⇄ act            act → reason is the ReAct loop-back
                        ↓
                   decide / halt
                        ↓
        page_on_call (⏸ interrupted before) / notify → END
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from react_langgraph.deps import NodeDeps
from react_langgraph.nodes import (
    make_act_node,
    make_decide_node,
    make_halt_node,
    make_notify_node,
    make_page_node,
    make_reason_node,
)
from react_langgraph.routing import route_after_decide, route_after_reason
from react_langgraph.state import TriageState
from triage_core.domain.models import (
    Action,
    Incident,
    IncidentMatches,
    Severity,
    SeverityResult,
    StopReason,
    TraceStep,
    TriageDecision,
)

#: The node execution is interrupted *before*. Paging a human is the sensitive step — the
#: equivalent of "see physio" in the training deck (``task02.md`` §5).
SENSITIVE_NODE = "page_on_call"

#: Extra super-steps of headroom over our own iteration cap. LangGraph enforces its own
#: ``recursion_limit`` and raises ``GraphRecursionError`` when it trips — which would bypass
#: the ``halt`` node entirely and crash instead of returning a partial result. Each iteration
#: costs two super-steps (reason + act), so we give the graph room for our cap to fire first.
_RECURSION_HEADROOM = 6

#: Domain types the checkpointer must be able to restore. LangGraph 1.x deserialises unknown
#: types with a warning today and refuses them in a future version, so the allowlist is
#: declared explicitly rather than relying on the permissive default.
CHECKPOINTED_TYPES: tuple[type, ...] = (
    Action,
    Incident,
    IncidentMatches,
    Severity,
    SeverityResult,
    StopReason,
    TraceStep,
    TriageDecision,
)


def build_checkpointer() -> BaseCheckpointSaver[str]:
    """Create the checkpointer, taught how to restore our domain objects.

    ``MemorySaver`` is per-process, which is enough for the assignment and matches the training
    deck. Swapping in a durable saver — so a pause survives across commands — is a one-line
    change here; nothing else in the graph depends on which saver is used.

    Returns:
        A checkpointer whose serializer allows this project's domain types.
    """
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES))


def recursion_limit(max_iterations: int) -> int:
    """The ``recursion_limit`` to pass in the run config.

    Args:
        max_iterations: Our own cap.

    Returns:
        A limit high enough that ``halt`` always fires before LangGraph's backstop.
    """
    return 2 * max_iterations + _RECURSION_HEADROOM


def build_graph(deps: NodeDeps) -> StateGraph[TriageState, None, TriageState, TriageState]:
    """Wire the nodes and edges.

    Both conditional edges are given an explicit path map, so the diagram renders correctly and
    a typo'd route name fails loudly instead of becoming a silent dead end.

    Args:
        deps: Injected collaborators for the nodes.

    Returns:
        The uncompiled graph.
    """
    graph: StateGraph[TriageState, None, TriageState, TriageState] = StateGraph(TriageState)

    graph.add_node("reason", make_reason_node(deps))
    graph.add_node("act", make_act_node(deps))
    graph.add_node("decide", make_decide_node(deps))
    graph.add_node(SENSITIVE_NODE, make_page_node(deps))
    graph.add_node("notify", make_notify_node(deps))
    graph.add_node("halt", make_halt_node(deps))

    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason",
        partial(route_after_reason, max_iterations=deps.settings.max_iterations),
        {"act": "act", "decide": "decide", "halt": "halt", "reason": "reason"},
    )

    # The loop-back edge — the part the assignment explicitly asks for. This single line is
    # what Part 1 expresses as `while ...: ... continue`.
    graph.add_edge("act", "reason")

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {SENSITIVE_NODE: SENSITIVE_NODE, "notify": "notify"},
    )
    graph.add_edge(SENSITIVE_NODE, END)
    graph.add_edge("notify", END)
    graph.add_edge("halt", END)
    return graph


def build_app(
    deps: NodeDeps,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[TriageState, None, TriageState, TriageState]:
    """Compile the graph with a checkpointer and the human-in-the-loop interrupt.

    The checkpointer is what makes the pause durable: state is saved under ``thread_id``, so a
    paused run can be inspected and resumed by a later invocation rather than blocking a
    function call.

    Args:
        deps: Injected collaborators.
        checkpointer: Defaults to a fresh :class:`MemorySaver`.

    Returns:
        The compiled app, which pauses before the sensitive node.
    """
    return build_graph(deps).compile(
        checkpointer=checkpointer if checkpointer is not None else build_checkpointer(),
        interrupt_before=[SENSITIVE_NODE],
    )


def render_mermaid(app: CompiledStateGraph[TriageState, None, TriageState, TriageState]) -> str:
    """Export the compiled graph as a mermaid diagram (``task02.md`` §7)."""
    return app.get_graph().draw_mermaid()


def render_ascii(app: CompiledStateGraph[TriageState, None, TriageState, TriageState]) -> str:
    """Export the compiled graph as ASCII art, if the optional renderer is available.

    LangGraph's ASCII renderer needs ``grandalf``, which is outside the dependency list this
    project is allowed (``CLAUDE.md`` §7). Mermaid is the committed diagram; this degrades to a
    note rather than failing the command.

    Returns:
        The ASCII diagram, or an explanatory line when ``grandalf`` is not installed.
    """
    try:
        return app.get_graph().draw_ascii()
    except ImportError:
        return (
            "(ASCII diagram needs the optional 'grandalf' package, which is outside this "
            "project's allowed dependencies — see the mermaid diagram above and docs/graph.md.)"
        )
