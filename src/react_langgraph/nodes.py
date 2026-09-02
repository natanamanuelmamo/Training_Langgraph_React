"""The graph's nodes (``task02.md`` §3).

Part 2 only — and orchestration only. Every tool, prompt and policy here is *imported* from
``triage_core``; nothing is reimplemented. If tool logic appeared in this file the submission
would be wrong (``task02.md`` §1), and ``tests/test_parity.py`` would go red.

Each node is built by a factory closing over :class:`NodeDeps`, because LangGraph requires the
``(state) -> partial update`` shape and that leaves nowhere to pass collaborators.

Every node returns a **partial** update. ``messages`` is appended to via its reducer; counters
are written as absolute values read off the cache so there is one source of truth.
"""

from __future__ import annotations

from typing import Any, Protocol

from react_langgraph.deps import CACHEABLE_TOOLS, NodeDeps
from react_langgraph.state import TriageState, observations, transcript_steps
from triage_core.domain.errors import ParseError, ToolExecutionError, ToolInputError
from triage_core.domain.models import (
    Action,
    Incident,
    IncidentMatches,
    Severity,
    SeverityResult,
    ToolResult,
    TraceStep,
)
from triage_core.domain.outcomes import (
    decision_from_payload,
    partial_decision,
    rejected_decision,
)
from triage_core.domain.prompts import SCRATCHPAD_HEADER
from triage_core.infra.tracing import render_transcript
from triage_core.llm.parsing import parse_react_step
from triage_core.tools.registry import describe_tools


class Node(Protocol):
    """A graph node: takes the state, returns a **partial** update.

    Declared as a Protocol rather than a ``Callable`` alias because LangGraph's own node
    protocol names its parameter ``state``, and a bare ``Callable`` erases parameter names —
    which makes every ``add_node`` call fail to type-check.
    """

    def __call__(self, state: TriageState) -> dict[str, Any]:
        """Run the node."""
        ...


def make_reason_node(deps: NodeDeps) -> Node:
    """Build the ``reason`` node: one LLM call over the accumulated transcript.

    The transcript is rendered with the same ``render_transcript`` and header that Part 1's
    ``Scratchpad.render`` uses, so both parts send the model byte-identical prompts. That is
    what makes the parity test meaningful rather than coincidental.

    Args:
        deps: Injected collaborators.

    Returns:
        The node function.
    """
    system_prompt = describe_tools(deps.tools)

    def reason(state: TriageState) -> dict[str, Any]:
        iteration = state["iteration"] + 1
        transcript = (
            SCRATCHPAD_HEADER + "\n" + render_transcript(state["log_line"], transcript_steps(state))
        )
        raw = deps.llm.complete(system_prompt, transcript)

        base: dict[str, Any] = {
            "iteration": iteration,
            "llm_calls": state["llm_calls"] + 1,
            "is_final": False,
        }

        try:
            step = parse_react_step(raw)
        except ParseError as exc:
            # Malformed output is a correctable mistake. Record it as an Observation and let
            # `route_after_reason` send us back round — the same recovery Part 1's loop does
            # with `continue`.
            return base | {
                "next_action": "",
                "messages": [
                    _node_row(iteration, "reason", "unparseable output → retry"),
                    TraceStep(
                        index=iteration,
                        kind="observation",
                        text=f"Output format error: {exc}",
                    ),
                ],
            }

        if step.is_final:
            return base | {
                "is_final": True,
                "final_payload": step.final_answer or {},
                "next_action": "",
                "messages": [
                    _node_row(iteration, "reason", f"Thought: {step.thought} → decide"),
                    TraceStep(index=iteration, kind="thought", text=step.thought),
                ],
            }

        return base | {
            "next_action": step.action or "",
            "next_action_input": step.action_input or {},
            "messages": [
                _node_row(iteration, "reason", f"Thought: {step.thought} → act"),
                TraceStep(index=iteration, kind="thought", text=step.thought),
            ],
        }

    return reason


def make_act_node(deps: NodeDeps) -> Node:
    """Build the ``act`` node: dispatch to the shared tool registry.

    Unknown tools and tool errors become Observations rather than exceptions, so the loop-back
    edge carries the correction to the next reasoning turn (Rule R10).

    Args:
        deps: Injected collaborators — note the registry is the same dict Part 1 dispatches
            through.

    Returns:
        The node function.
    """

    def act(state: TriageState) -> dict[str, Any]:
        iteration = state["iteration"]
        name = state.get("next_action", "")
        args = state.get("next_action_input") or {}

        tool = deps.tools.get(name)
        if tool is None:
            text = f"Unknown tool {name!r}. Available tools: {sorted(deps.tools)}"
            return {
                "next_action": "",
                "messages": [
                    _node_row(iteration, "act", f"unknown tool {name!r}"),
                    TraceStep(index=iteration, kind="observation", text=text),
                ],
            }

        observation, verdict, payload = _invoke(deps, tool.name, args)

        update: dict[str, Any] = {
            "next_action": "",
            "tool_calls": state["tool_calls"] + 1,
            "cache_hits": deps.cache.stats.hits,
            "cache_misses": deps.cache.stats.misses,
            "messages": [
                _node_row(
                    iteration,
                    "act",
                    f"{tool.name}{'' if verdict is None else f'  CACHE {verdict.upper()}'}"
                    f" → {_summarise(observation)}",
                ),
                TraceStep(
                    index=iteration,
                    kind="action",
                    label=tool.name,
                    text=_dump(args),
                ),
                TraceStep(
                    index=iteration,
                    kind="observation",
                    text=observation,
                    cache=verdict,  # type: ignore[arg-type]
                ),
            ],
        }
        update |= _structured_outputs(payload)
        return update

    return act


def make_decide_node(deps: NodeDeps) -> Node:
    """Build the ``decide`` node: turn the Final Answer into a decision.

    Sets ``requires_approval`` for the sensitive action, which is what
    :func:`~react_langgraph.routing.route_after_decide` reads.

    Args:
        deps: Injected collaborators.

    Returns:
        The node function.
    """

    def decide(state: TriageState) -> dict[str, Any]:
        iteration = state["iteration"]
        try:
            decision = decision_from_payload(state.get("final_payload") or {})
        except (KeyError, TypeError, ValueError) as exc:
            # The parser already validated shape, so this is rare. Fall back to the same
            # partial result `halt` would produce rather than crashing the graph.
            decision = partial_decision(observations(state), deps.settings.max_iterations)
            return {
                "decision": decision,
                "requires_approval": False,
                "messages": [
                    _node_row(iteration, "decide", f"unusable final answer ({exc}) → partial"),
                    TraceStep(
                        index=iteration,
                        kind="notice",
                        text=f"Final Answer could not be used: {exc}",
                    ),
                ],
            }

        requires_approval = decision.action is Action.PAGE_ON_CALL
        return {
            "decision": decision,
            "requires_approval": requires_approval,
            "messages": [
                _node_row(
                    iteration,
                    "decide",
                    f"action={decision.action}  requires_approval={requires_approval}",
                ),
                TraceStep(index=iteration, kind="final", text=str(decision.action)),
            ],
        }

    return decide


def make_page_node(deps: NodeDeps) -> Node:
    """Build the ``page_on_call`` node — **the sensitive step**.

    Execution is interrupted *before* this node by ``interrupt_before=["page_on_call"]``, so
    it only ever runs after a human has approved. Nobody gets woken at 3am otherwise.

    Args:
        deps: Injected collaborators; ``page_spy`` counts executions for the tests.

    Returns:
        The node function.
    """

    def page_on_call(state: TriageState) -> dict[str, Any]:
        deps.page_spy.increment()
        return {
            "approval": state.get("approval", "approved"),
            "messages": [
                _node_row(state["iteration"], "page_on_call", f"paged: {deps.on_call_target}")
            ],
        }

    return page_on_call


def make_notify_node(deps: NodeDeps) -> Node:
    """Build the ``notify`` node — the non-sensitive terminal path.

    Also the landing point for a rejected page. Rejection must change the *outcome*, not just
    the path, so the decision is rewritten here (``task02.md`` §5.5).

    Args:
        deps: Injected collaborators.

    Returns:
        The node function.
    """

    def notify(state: TriageState) -> dict[str, Any]:
        iteration = state["iteration"]
        decision = state.get("decision")

        if state.get("approval") == "rejected" and decision is not None:
            rewritten = rejected_decision(decision, state.get("approval_note", ""))
            return {
                "decision": rewritten,
                "messages": [
                    _node_row(
                        iteration,
                        "notify",
                        f"page rejected by human → {rewritten.action}",
                    )
                ],
            }

        action = decision.action if decision is not None else Action.FILE_TICKET
        verb = "ticket filed" if action is Action.FILE_TICKET else "recorded, no action"
        return {"messages": [_node_row(iteration, "notify", verb)]}

    return notify


def make_halt_node(deps: NodeDeps) -> Node:
    """Build the ``halt`` node: the iteration cap reached.

    Produces the same partial decision Part 1's loop returns in the same situation, because
    both call ``domain.outcomes.partial_decision`` (Rule R10, Rule R3).

    Args:
        deps: Injected collaborators.

    Returns:
        The node function.
    """

    def halt(state: TriageState) -> dict[str, Any]:
        decision = partial_decision(observations(state), deps.settings.max_iterations)
        return {
            "decision": decision,
            "requires_approval": False,
            "messages": [
                _node_row(
                    state["iteration"],
                    "halt",
                    f"iteration limit {deps.settings.max_iterations} reached → partial result",
                )
            ],
        }

    return halt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _invoke(deps: NodeDeps, name: str, args: dict[str, Any]) -> tuple[str, str | None, Any]:
    """Run one tool through the cache, converting declared failures into Observation text.

    Mirrors Part 1's ``_invoke``: only ``ToolInputError`` and ``ToolExecutionError`` are
    absorbed. Anything else is a bug in a tool and is left to surface.

    Returns:
        The Observation text, the cache verdict, and the structured payload (``None`` on
        failure).
    """
    tool = deps.tools[name]
    try:
        if name in CACHEABLE_TOOLS:
            outcome = deps.cache.get_or_compute(name, args, lambda: tool.run(**args))
            result: ToolResult[Any] = outcome.value
            return result.render(), ("hit" if outcome.hit else "miss"), result.payload
        result = tool.run(**args)
        return result.render(), None, result.payload
    except ToolInputError as exc:
        return f"Tool error: {exc}", None, None
    except ToolExecutionError as exc:
        return f"Tool failed: {exc}", None, None


def _structured_outputs(payload: Any) -> dict[str, Any]:
    """Lift a tool's typed result into the dedicated state fields.

    Keeps ``severity`` / ``signature`` / ``incidents`` queryable without re-parsing the
    transcript.
    """
    if isinstance(payload, SeverityResult):
        return {"severity": payload.severity, "signature": payload.signature}
    if isinstance(payload, IncidentMatches):
        return {"incidents": list(payload.matches)}
    return {}


def _node_row(index: int, label: str, text: str) -> TraceStep:
    """A display-only trace row naming the node that just ran."""
    return TraceStep(index=index, kind="node", label=label, text=text)


def _dump(args: dict[str, Any]) -> str:
    """Render an Action Input for the transcript, matching Part 1's formatting."""
    import json

    return json.dumps(args, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    """Serialise domain objects that appear inside an Action Input."""
    if isinstance(value, Incident):
        return value.to_dict()
    if isinstance(value, (Severity, Action)):
        return value.value
    return str(value)


def _summarise(observation: str) -> str:
    """Shorten an Observation to one line for the node row."""
    first = observation.splitlines()[0] if observation else ""
    return first if len(first) <= 70 else first[:67] + "..."
