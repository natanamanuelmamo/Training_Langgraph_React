"""Injected dependencies for the graph's nodes (``task_2_implementation.md`` §9.1).

Part 2 only. LangGraph nodes must have the shape ``(state) -> partial update``, which leaves
nowhere to pass collaborators. Node factories in ``nodes.py`` close over one :class:`NodeDeps`
instead, so the LLM, the tool registry and the cache are all injected rather than global —
``task02.md`` §4 requires that of the cache, and the same argument applies to the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from triage_core.config import Settings
from triage_core.domain.models import Counter
from triage_core.domain.ports import TextCompleter
from triage_core.infra.cache import ResultCache
from triage_core.tools.base import Tool

#: Tools worth memoising — the same choice Part 1 makes, for the same reason.
CACHEABLE_TOOLS = frozenset({"lookup_incidents"})


@dataclass(frozen=True, slots=True)
class NodeDeps:
    """Everything the nodes need, supplied once when the graph is built."""

    llm: TextCompleter
    tools: Mapping[str, Tool]
    cache: ResultCache
    settings: Settings

    #: Counts executions of the sensitive node. Tests assert this is exactly zero after a
    #: rejection — checking the final state alone would not catch a node that ran and was
    #: then overwritten.
    page_spy: Counter = field(default_factory=Counter)

    #: On-call rota the page would go to. Printed, never actually dispatched.
    on_call_target: str = "sre-primary"
