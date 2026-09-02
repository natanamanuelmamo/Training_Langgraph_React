"""Command-line entry point for Part 1 (``task01.md`` §5).

Part 1 only. Prints the full reasoning trace — every iteration's Thought, Action, Action Input
and Observation — then the result block and the counters line. The trace is a deliverable, not
debug output (Rule R8).

Usage::

    python -m react_from_scratch.cli "2026-08-29T03:14:07Z ERROR payments-api ..."
    python -m react_from_scratch.cli --file data/samples/pool_exhausted.log
    python -m react_from_scratch.cli --fake --max-iters 4
    python -m react_from_scratch.cli --file data/samples/slow_query.log --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from react_from_scratch.agent import run_react_agent
from triage_core.config import build_llm, load_settings
from triage_core.domain.errors import DataError
from triage_core.infra.cache import ResultCache
from triage_core.infra.tracing import Tracer, render_summary, render_trace
from triage_core.llm.client import LLMError
from triage_core.tools.registry import build_registry

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m react_from_scratch.cli",
        description="Triage one log line with a hand-written ReAct loop (Part 1).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("log_line", nargs="?", help="the raw log line to triage")
    source.add_argument("--file", type=Path, help="read the log line from a file instead")

    parser.add_argument(
        "--fake", action="store_true", help="force the offline FakeLLM even if a key is set"
    )
    parser.add_argument(
        "--max-iters", type=int, default=None, help="override the iteration cap (default 6)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="disable the result cache to compare traces"
    )
    parser.add_argument(
        "--incidents", type=Path, default=None, help="override the incident history path"
    )
    parser.add_argument("--json", action="store_true", help="print the run as JSON instead")
    return parser


def _read_log_line(args: argparse.Namespace) -> str:
    """Resolve the log line from the positional argument or ``--file``.

    Raises:
        ValueError: The file is empty or holds only blank lines.
    """
    if args.file is not None:
        path: Path = args.file
        text: str = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        raise ValueError(f"{path} contains no non-blank line")
    return str(args.log_line).strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``. Passed explicitly by tests.

    Returns:
        ``0`` on a complete decision, ``1`` on a partial one, ``2`` on a usage or data error.
    """
    args = build_parser().parse_args(argv)

    try:
        log_line = _read_log_line(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    overrides = {}
    if args.max_iters is not None:
        overrides["max_iterations"] = args.max_iters
    if args.incidents is not None:
        overrides["incidents_path"] = args.incidents
    if args.no_cache:
        overrides["cache_enabled"] = False
    settings = load_settings(**overrides)

    llm = build_llm(settings, force_fake=args.fake)

    try:
        tools = build_registry(
            llm=llm,
            incidents_path=settings.incidents_path,
            lookback_days=settings.lookback_days,
            today=settings.today,
        )
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    cache = ResultCache(enabled=settings.cache_enabled)
    tracer = Tracer()

    try:
        run = run_react_agent(
            log_line,
            llm=llm,
            tools=tools,
            cache=cache,
            tracer=tracer,
            max_iterations=settings.max_iterations,
        )
    except LLMError as exc:
        # A provider misconfiguration (bad key, unknown model, network down) — report it
        # cleanly rather than dumping a traceback.
        print(f"error: LLM request failed — {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        print(json.dumps(run.to_dict(), indent=2))
    else:
        print()
        print(render_trace(run.trace, title="ReAct (raw Python)", log_line=log_line))
        print()
        print(render_summary(run))

    return EXIT_OK if run.decision.complete else EXIT_PARTIAL


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
