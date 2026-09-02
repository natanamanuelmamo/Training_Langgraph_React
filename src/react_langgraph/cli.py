"""Command-line entry point for Part 2 (``task02.md`` §6).

Part 2 only. Prints node transitions, the loop-backs, the cache verdicts, the pause and the
resume — the trace is the evidence the pattern works (Rule R8).

Usage::

    python -m react_langgraph.cli "2026-08-29T03:14:07Z ERROR payments-api ..."
    python -m react_langgraph.cli --thread-id inc-001 --auto-approve
    python -m react_langgraph.cli --thread-id inc-001 --auto-reject --note "known noisy host"
    python -m react_langgraph.cli --demo-cache
    python -m react_langgraph.cli --resume inc-001
    python -m react_langgraph.cli --print-graph
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from react_langgraph.builder import build_app
from react_langgraph.deps import NodeDeps
from react_langgraph.hitl import (
    App,
    RunConfig,
    approve,
    is_paused,
    make_config,
    pending_state,
    pending_summary,
    prompt_for_approval,
    reject,
)
from react_langgraph.state import TriageState, initial_state
from triage_core.config import Settings, build_llm, load_settings
from triage_core.domain.errors import DataError
from triage_core.domain.models import TriageRun
from triage_core.infra.cache import ResultCache
from triage_core.infra.tracing import render_node_trace, render_summary
from triage_core.llm.client import LLMError
from triage_core.tools.registry import build_registry

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_PAUSED = 3

DEFAULT_SAMPLE = Path("data/samples/pool_exhausted.log")


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m react_langgraph.cli",
        description="Triage one log line with a LangGraph ReAct agent (Part 2).",
    )
    parser.add_argument("log_line", nargs="?", help="the raw log line to triage")
    parser.add_argument("--file", type=Path, help="read the log line from a file instead")
    parser.add_argument(
        "--thread-id", default="inc-001", help="checkpointer key for this run (default inc-001)"
    )

    approval = parser.add_mutually_exclusive_group()
    approval.add_argument(
        "--auto-approve", action="store_true", help="approve the page without prompting"
    )
    approval.add_argument(
        "--auto-reject", action="store_true", help="reject the page without prompting"
    )
    parser.add_argument("--note", default="", help="reason recorded with --auto-reject")

    parser.add_argument(
        "--demo-cache",
        action="store_true",
        help="triage the same log twice in one process and print cache hit/miss",
    )
    parser.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help=(
            "run in two explicit phases — pause, inspect the saved state, then resume it as a "
            "separate invocation. Shows that the pause is persisted state rather than a "
            "blocked call. Note: MemorySaver is per-process, so resuming in a *later* command "
            "needs a durable checkpointer (see --print-graph notes and the README)"
        ),
    )
    parser.add_argument(
        "--print-graph", action="store_true", help="print the compiled graph and exit"
    )

    parser.add_argument("--fake", action="store_true", help="force the offline FakeLLM")
    parser.add_argument("--no-cache", action="store_true", help="disable the result cache")
    parser.add_argument("--max-iters", type=int, default=None, help="override the iteration cap")
    parser.add_argument("--incidents", type=Path, default=None, help="incident history path")
    parser.add_argument("--json", action="store_true", help="print the run as JSON instead")
    return parser


def _read_log_line(args: argparse.Namespace) -> str:
    """Resolve the log line from ``--file``, the positional argument, or the default sample.

    Raises:
        ValueError: The file holds only blank lines.
    """
    if args.file is not None:
        path: Path = args.file
        text: str = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        raise ValueError(f"{path} contains no non-blank line")
    if args.log_line:
        return str(args.log_line).strip()
    return DEFAULT_SAMPLE.read_text(encoding="utf-8").strip().splitlines()[0]


def _build_deps(args: argparse.Namespace, settings: Settings) -> NodeDeps:
    """Assemble the injected collaborators for one process."""
    llm = build_llm(settings, force_fake=args.fake)
    tools = build_registry(
        llm=llm,
        incidents_path=settings.incidents_path,
        lookback_days=settings.lookback_days,
        today=settings.today,
    )
    return NodeDeps(
        llm=llm,
        tools=tools,
        cache=ResultCache(enabled=settings.cache_enabled),
        settings=settings,
    )


def _settings_from(args: argparse.Namespace) -> Settings:
    """Apply CLI flags on top of the environment."""
    overrides: dict[str, object] = {}
    if args.max_iters is not None:
        overrides["max_iterations"] = args.max_iters
    if args.incidents is not None:
        overrides["incidents_path"] = args.incidents
    if args.no_cache:
        overrides["cache_enabled"] = False
    return load_settings(**overrides)


def _as_run(state: TriageState, deps: NodeDeps) -> TriageRun:
    """Reshape the final graph state into the same result object Part 1 returns."""
    decision = state.get("decision")
    assert decision is not None, "every terminal path sets a decision"
    return TriageRun(
        decision=decision,
        trace=tuple(state["messages"]),
        iterations=state["iteration"],
        tool_calls=state["tool_calls"],
        llm_calls=state["llm_calls"],
        cache_hits=deps.cache.stats.hits,
        cache_misses=deps.cache.stats.misses,
    )


def _report(state: TriageState, deps: NodeDeps, args: argparse.Namespace) -> int:
    """Print the trace and result, and pick the exit code."""
    run = _as_run(state, deps)
    if args.json:
        print(json.dumps(run.to_dict(), indent=2))
    else:
        print()
        print(render_node_trace(run.trace, title="LangGraph ReAct", thread_id=state["thread_id"]))
        print()
        print(render_summary(run))
    return EXIT_OK if run.decision.complete else EXIT_PARTIAL


def _narrate(args: argparse.Namespace) -> TextIO:
    """Where human-facing narration goes.

    Under ``--json`` stdout carries machine-readable output only, so the pause banner and the
    approval summary move to stderr rather than corrupting the document a caller is piping.
    """
    return sys.stderr if args.json else sys.stdout


def _settle(app: App, config: RunConfig, args: argparse.Namespace) -> TriageState | None:
    """Handle the pause, if one happened.

    Returns:
        The final state, or ``None`` when the run is left paused for a later ``--resume``.
    """
    if not is_paused(app, config):
        return pending_state(app, config)

    state = pending_state(app, config)
    summary = pending_summary(state)
    out = _narrate(args)

    if args.auto_approve:
        print("\n⏸  PAUSED before node 'page_on_call' — auto-approving\n" + summary, file=out)
        return approve(app, config)
    if args.auto_reject:
        print("\n⏸  PAUSED before node 'page_on_call' — auto-rejecting\n" + summary, file=out)
        return reject(app, config, args.note or "rejected via --auto-reject")

    print("\n⏸  PAUSED before node 'page_on_call'", file=out)
    if prompt_for_approval(summary):
        return approve(app, config)
    return reject(app, config, args.note or "declined at the prompt")


def _run_once(args: argparse.Namespace, settings: Settings) -> int:
    """Triage one log line, honouring the pause."""
    log_line = _read_log_line(args)
    deps = _build_deps(args, settings)
    app = build_app(deps)
    config = make_config(args.thread_id, settings.max_iterations)

    app.invoke(initial_state(log_line, args.thread_id), config=config)
    state = _settle(app, config, args)
    if state is None:  # pragma: no cover - only reachable via a future non-blocking mode
        return EXIT_PAUSED
    return _report(state, deps, args)


def _run_resume(args: argparse.Namespace, settings: Settings) -> int:
    """Pause, inspect the persisted state, then resume as a separate invocation.

    This is what the checkpointer buys: the pause is saved state keyed by ``thread_id``, not a
    blocked function call. The two ``invoke`` calls below are genuinely independent — the
    second one is handed ``None`` as input and reconstructs everything from the checkpoint.

    ``MemorySaver`` lives in this process, so resuming from a *later* command needs a durable
    saver. ``build_app`` already takes a ``checkpointer`` argument, so that is a one-line swap.
    """
    thread_id = str(args.resume)
    log_line = _read_log_line(args)
    deps = _build_deps(args, settings)
    app = build_app(deps)
    config = make_config(thread_id, settings.max_iterations)
    out = _narrate(args)

    print(f"\n[phase 1] invoke thread {thread_id!r}", file=out)
    app.invoke(initial_state(log_line, thread_id), config=config)

    snapshot = app.get_state(config)
    print(f"          app.get_state(config).next = {snapshot.next}", file=out)

    if not snapshot.next:
        print("          run reached END without pausing — nothing to resume", file=out)
        return _report(pending_state(app, config), deps, args)

    print(f"\n[phase 2] resume thread {thread_id!r} from the saved checkpoint", file=out)
    state = _settle(app, config, args)
    assert state is not None
    print(f"          app.get_state(config).next = {app.get_state(config).next}", file=out)
    return _report(state, deps, args)


def _run_demo_cache(args: argparse.Namespace, settings: Settings) -> int:
    """Triage the same log twice in one process to show the cache working (``task02.md`` §4)."""
    log_line = _read_log_line(args)
    deps = _build_deps(args, settings)
    app = build_app(deps)

    print("\n── Cache demo " + "─" * 45)
    print(f"Input: {log_line}\n")

    for run_number in (1, 2):
        before_hits = deps.cache.stats.hits
        config = make_config(f"{args.thread_id}-demo-{run_number}", settings.max_iterations)
        app.invoke(initial_state(log_line, config["configurable"]["thread_id"]), config=config)
        if is_paused(app, config):
            approve(app, config)
        verdict = "HIT " if deps.cache.stats.hits > before_hits else "MISS"
        print(f"run {run_number}  lookup_incidents  {verdict}")

    stats = deps.cache.stats
    print(f"\ncache: {stats.render()}")
    if not deps.cache.enabled:
        print("(--no-cache given: every lookup is a miss, and the tool body runs every time)")
    return EXIT_OK


def _run_print_graph(args: argparse.Namespace, settings: Settings) -> int:
    """Print the compiled graph as mermaid and ASCII (``task02.md`` §6)."""
    deps = _build_deps(args, settings)
    app = build_app(deps)
    from react_langgraph.builder import render_ascii, render_mermaid

    print("```mermaid")
    print(render_mermaid(app).rstrip())
    print("```")
    print()
    print(render_ascii(app))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` complete, ``1`` partial, ``2`` usage or data error, ``3`` still paused.
    """
    args = build_parser().parse_args(argv)
    settings = _settings_from(args)

    try:
        if args.print_graph:
            return _run_print_graph(args, settings)
        if args.demo_cache:
            return _run_demo_cache(args, settings)
        if args.resume:
            return _run_resume(args, settings)
        return _run_once(args, settings)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except LLMError as exc:
        # Provider misconfiguration (bad key, unknown model, network down) — clean message,
        # not a traceback.
        print(f"error: LLM request failed — {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
