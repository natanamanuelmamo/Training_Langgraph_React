"""THE ReAct loop, written by hand in raw Python (``task01.md`` §3).

Part 1 of the assignment. No LangChain, no LangGraph, no agent framework of any kind — the
Anthropic SDK is the only external dependency anywhere in this import graph, and
``tests/test_isolation.py`` proves it (Rule R1).

The control flow is deliberately literal (Rule R2):

* :func:`run_react_agent` contains an actual ``while`` loop, not recursion or a generator.
* Tools are an actual ``dict`` of callables, dispatched with ``tools.get(name)``.
* The reason step is an LLM call this module makes, whose output this module parses.

The four phases are marked with ``# ---- REASON/TERMINATE/ACT/OBSERVE ----`` banners so a
reviewer can point at the exact line where each happens.

Every exit path returns a well-formed :class:`TriageRun`: a final answer, the iteration cap, or
an unrecoverable error. Unknown tools and malformed model output become Observations fed back
into the loop rather than exceptions (Rule R10).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from react_from_scratch.scratchpad import Scratchpad
from triage_core.domain.errors import ParseError, ToolExecutionError, ToolInputError
from triage_core.domain.models import (
    ReActStep,
    RunCounters,
    ToolResult,
    TraceStep,
    TriageDecision,
    TriageRun,
)
from triage_core.domain.outcomes import decision_from_payload, partial_decision
from triage_core.domain.ports import TextCompleter
from triage_core.infra.cache import ResultCache
from triage_core.infra.tracing import Tracer
from triage_core.llm.parsing import parse_react_step
from triage_core.tools.base import Tool
from triage_core.tools.registry import describe_tools

#: Tools whose results are worth memoising. ``lookup_incidents`` is deterministic and gets
#: called repeatedly across runs — the same choice Part 2 makes (``task02.md`` §4).
CACHEABLE_TOOLS = frozenset({"lookup_incidents"})


def run_react_agent(
    log_line: str,
    *,
    llm: TextCompleter,
    tools: Mapping[str, Tool],
    cache: ResultCache | None = None,
    tracer: Tracer | None = None,
    max_iterations: int = 6,
) -> TriageRun:
    """Run the ReAct loop until the agent finishes or runs out of iterations.

    Args:
        log_line: The raw log line to triage.
        llm: The reasoning back end.
        tools: Name-to-tool mapping — a real dict, dispatched by name (Rule R2).
        cache: Shared result cache. A fresh disabled one is used when omitted.
        tracer: Trace collector. A fresh one is used when omitted.
        max_iterations: Hard cap on reasoning turns (Rule R10).

    Returns:
        The decision plus the full trace and counters, whichever way the loop ended.
    """
    cache = cache if cache is not None else ResultCache(enabled=False)
    tracer = tracer if tracer is not None else Tracer()
    counters = RunCounters()

    scratchpad = Scratchpad(log_line)
    system_prompt = describe_tools(tools)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        counters.iterations = iteration
        scratchpad.begin_iteration(iteration)

        # ---- REASON -------------------------------------------------------
        raw = llm.complete(system_prompt, scratchpad.render())
        counters.llm_calls += 1

        try:
            step = parse_react_step(raw)
        except ParseError as exc:
            # Malformed output is a correctable mistake, not a crash. Feeding the parser's
            # message back verbatim is the model's only chance to fix itself.
            _observe(scratchpad, tracer, f"Output format error: {exc}")
            continue

        _record(tracer, scratchpad.add_thought(step.thought))

        # ---- TERMINATE ----------------------------------------------------
        if step.is_final:
            decision = _build_decision(step, scratchpad)
            if decision is not None:
                _record(tracer, scratchpad.add_final(str(decision.action)))
                return _finish(decision, tracer, counters, cache)
            continue

        # ---- ACT ----------------------------------------------------------
        tool = tools.get(step.action or "")
        if tool is None:
            _observe(
                scratchpad,
                tracer,
                f"Unknown tool {step.action!r}. Available tools: {sorted(tools)}",
            )
            continue

        args = step.action_input or {}
        _record(tracer, scratchpad.add_action(tool.name, args))
        counters.tool_calls += 1

        observation, verdict = _invoke(tool, args, cache)

        # ---- OBSERVE ------------------------------------------------------
        _observe(scratchpad, tracer, observation, cache=verdict)

    return _finish(
        partial_decision(scratchpad.observations, max_iterations), tracer, counters, cache
    )


def _invoke(tool: Tool, args: Mapping[str, Any], cache: ResultCache) -> tuple[str, str | None]:
    """Call one tool, converting every expected failure into Observation text.

    Args:
        tool: The tool to run.
        args: Its parsed arguments.
        cache: The shared cache; consulted only for :data:`CACHEABLE_TOOLS`.

    Returns:
        The Observation text and the cache verdict (``"hit"``, ``"miss"`` or ``None``).

    Note:
        Only the two declared failure classes are absorbed. Every tool validates its own
        arguments and raises ``ToolInputError``, so any *other* exception is a bug in a tool
        rather than a bad model argument — it is left to propagate instead of being disguised
        as an Observation the model is asked to work around.
    """
    try:
        if tool.name in CACHEABLE_TOOLS:
            outcome = cache.get_or_compute(tool.name, args, lambda: tool.run(**args))
            result: ToolResult[Any] = outcome.value
            return result.render(), ("hit" if outcome.hit else "miss")
        return tool.run(**args).render(), None
    except ToolInputError as exc:
        return f"Tool error: {exc}", None
    except ToolExecutionError as exc:
        return f"Tool failed: {exc}", None


def _build_decision(step: ReActStep, scratchpad: Scratchpad) -> TriageDecision | None:
    """Turn a validated Final Answer into a :class:`TriageDecision`.

    Conversion itself lives in ``domain.outcomes`` so Part 2's ``decide`` node produces an
    identical decision from an identical payload (Rule R3). If conversion fails the loop
    continues with an Observation rather than raising — a bad final answer is just another
    correctable mistake.

    Args:
        step: The parsed final step.
        scratchpad: The transcript, appended to when conversion fails.

    Returns:
        The decision, or ``None`` when the payload could not be converted.
    """
    try:
        return decision_from_payload(step.final_answer or {})
    except (KeyError, TypeError, ValueError) as exc:
        scratchpad.add_observation(f"Final Answer could not be used: {exc}")
        return None


def _observe(
    scratchpad: Scratchpad, tracer: Tracer, text: str, *, cache: str | None = None
) -> None:
    """Append an Observation to both the transcript and the trace."""
    _record(tracer, scratchpad.add_observation(text, cache=cache))


def _record(tracer: Tracer, step: TraceStep) -> None:
    """Mirror a transcript step into the trace."""
    tracer.add(step)


def _finish(
    decision: TriageDecision, tracer: Tracer, counters: RunCounters, cache: ResultCache
) -> TriageRun:
    """Assemble the run result."""
    return TriageRun(
        decision=decision,
        trace=tracer.steps,
        iterations=counters.iterations,
        tool_calls=counters.tool_calls,
        llm_calls=counters.llm_calls,
        cache_hits=cache.stats.hits,
        cache_misses=cache.stats.misses,
    )
