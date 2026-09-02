"""Builders for the Part 2 tests.

Kept out of ``conftest.py`` so that importing the shared fixtures never pulls in LangGraph —
Part 1 must remain runnable and testable with the optional ``graph`` extra uninstalled.
Every graph test module guards its import with ``pytest.importorskip("langgraph")``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.conftest import TODAY

from react_langgraph.builder import build_app
from react_langgraph.deps import NodeDeps
from react_langgraph.hitl import App, RunConfig, make_config
from react_langgraph.state import TriageState, initial_state
from triage_core.config import Settings
from triage_core.domain.models import Counter
from triage_core.infra.cache import ResultCache
from triage_core.llm.client import FakeLLM, ScriptedLLM
from triage_core.tools.registry import build_registry


@dataclass
class Harness:
    """One compiled graph plus everything needed to drive and inspect it."""

    app: App
    deps: NodeDeps
    config: RunConfig
    thread_id: str

    @property
    def pages_sent(self) -> int:
        """How many times the sensitive node actually executed."""
        return self.deps.page_spy.value

    def start(self, log_line: str) -> TriageState:
        """Invoke the graph, returning whatever state it reaches or pauses at."""
        result: TriageState = self.app.invoke(
            initial_state(log_line, self.thread_id), config=self.config
        )
        return result

    def node_labels(self) -> list[str]:
        """The nodes that ran, in order, read from the persisted state."""
        values: TriageState = self.app.get_state(self.config).values
        return [step.label for step in values["messages"] if step.kind == "node"]


def make_harness(
    incidents_path: Path,
    *,
    responses: Sequence[str] | None = None,
    max_iterations: int = 6,
    cache_enabled: bool = True,
    thread_id: str = "test-thread",
    cache: ResultCache | None = None,
) -> Harness:
    """Build a compiled graph wired to deterministic, offline dependencies.

    Args:
        incidents_path: The test incident history.
        responses: A script for :class:`ScriptedLLM`; ``None`` uses the offline
            :class:`FakeLLM`, which drives a correct loop on any log line.
        max_iterations: The iteration cap under test.
        cache_enabled: Whether the shared cache stores entries.
        thread_id: Checkpointer key.
        cache: Reuse an existing cache across harnesses, for the hit/miss tests.

    Returns:
        The harness.
    """
    llm = ScriptedLLM(responses) if responses is not None else FakeLLM()
    settings = Settings(
        max_iterations=max_iterations,
        incidents_path=incidents_path,
        cache_enabled=cache_enabled,
        today=TODAY,
    )
    tools = build_registry(llm=llm, incidents_path=incidents_path, lookback_days=90, today=TODAY)
    deps = NodeDeps(
        llm=llm,
        tools=tools,
        cache=cache if cache is not None else ResultCache(enabled=cache_enabled),
        settings=settings,
        page_spy=Counter(),
    )
    return Harness(
        app=build_app(deps),
        deps=deps,
        config=make_config(thread_id, max_iterations),
        thread_id=thread_id,
    )
