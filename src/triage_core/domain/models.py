"""Core data shapes for the Log-Triage Agent.

Serves both parts of the assignment: every object crossing a module boundary is one of
these, never a bare dict (Rule R5). This module imports only the standard library
(Rule R6) — it is the bottom of the dependency graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar, cast

T = TypeVar("T")

TraceKind = Literal["thought", "action", "observation", "final", "node", "notice"]
CacheVerdict = Literal["hit", "miss"]

# StrEnum (not `str, Enum`) so f-strings render the bare value: the trace must read
# `action=file_ticket`, never `action=Action.FILE_TICKET`.


class Severity(StrEnum):
    """How bad a log line is."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Action(StrEnum):
    """What the agent decides to do about a log line."""

    PAGE_ON_CALL = "page_on_call"
    FILE_TICKET = "file_ticket"
    IGNORE = "ignore"


class StopReason(StrEnum):
    """Why a run ended."""

    FINAL_ANSWER = "final_answer"
    ITERATION_LIMIT = "iteration_limit"
    HUMAN_REJECTED = "human_rejected"


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One parsed log line."""

    raw: str
    timestamp: datetime | None
    level: str | None
    service: str | None
    message: str


@dataclass(frozen=True, slots=True)
class Incident:
    """A past incident recorded in ``data/incidents.json``."""

    incident_id: str
    signature: str
    severity: Severity
    opened_at: date
    resolved: bool
    resolution: str | None
    occurrences: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the LLM's ``Action Input`` round-trip and for ``--json`` output."""
        return {
            "incident_id": self.incident_id,
            "signature": self.signature,
            "severity": self.severity.value,
            "opened_at": self.opened_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "occurrences": self.occurrences,
        }


@dataclass(frozen=True, slots=True)
class SeverityResult:
    """Output of ``classify_severity``."""

    severity: Severity
    signature: str
    confidence: float
    matched_rule: str


@dataclass(frozen=True, slots=True)
class IncidentMatches:
    """Output of ``lookup_incidents``."""

    signature: str
    matches: tuple[Incident, ...]
    total_occurrences: int
    lookback_days: int
    latest: date | None

    def __post_init__(self) -> None:
        """Normalise ``matches`` to a tuple — see :meth:`TriageDecision.__post_init__`."""
        value: object = self.matches
        if not isinstance(value, tuple):
            object.__setattr__(self, "matches", tuple(cast("Iterable[Incident]", value)))


@dataclass(frozen=True, slots=True)
class ActionRecommendation:
    """Output of ``recommend_action`` — the escalation policy's verdict."""

    action: Action
    justification: str
    policy_rule: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ToolResult(Generic[T]):
    """What every tool returns — never a raw string (``task01.md`` §2).

    ``ok=False`` means the tool ran and could not produce an answer. A structurally invalid
    call raises :class:`~triage_core.domain.errors.ToolInputError` instead. Both become
    Observations; the split keeps "found nothing" distinguishable from "called wrong".
    """

    tool_name: str
    ok: bool
    payload: T | None = None
    error: str | None = None
    detail: str = ""

    def render(self) -> str:
        """Render as the Observation text the agent feeds back into the next reason step."""
        if not self.ok:
            return f"error: {self.error}"
        return self.detail


@dataclass(frozen=True, slots=True)
class ReActStep:
    """One parsed reasoning step: a Thought plus either an Action or a Final Answer."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    final_answer: dict[str, Any] | None = None

    @property
    def is_final(self) -> bool:
        """True when the model has concluded and emitted a Final Answer."""
        return self.final_answer is not None


@dataclass(frozen=True, slots=True)
class TriageDecision:
    """The agent's final output (``task01.md`` §1)."""

    action: Action
    severity: Severity
    confidence: float
    justification: str
    matched_incidents: tuple[str, ...] = ()
    complete: bool = True
    stop_reason: StopReason = StopReason.FINAL_ANSWER

    def __post_init__(self) -> None:
        """Normalise ``matched_incidents`` to a tuple.

        Part 2 round-trips this object through LangGraph's checkpointer, which restores every
        sequence as a ``list``. Without this coercion an otherwise-identical decision would
        compare unequal after a pause — and ``tests/test_parity.py``, which compares Part 1's
        decision with Part 2's, would fail for a reason that has nothing to do with the agents.

        The widening to ``object`` is deliberate: the annotation promises a tuple, so a direct
        ``isinstance`` check reads as always-true to a type checker, but deserialization does
        not honour annotations at runtime.
        """
        value: object = self.matched_incidents
        if not isinstance(value, tuple):
            object.__setattr__(self, "matched_incidents", tuple(cast("Iterable[str]", value)))

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``--json`` output."""
        return {
            "action": self.action.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "justification": self.justification,
            "matched_incidents": list(self.matched_incidents),
            "complete": self.complete,
            "stop_reason": self.stop_reason.value,
        }


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One line of the printed trace (Rule R8).

    Part 1 accumulates these in a ``Tracer``; Part 2 accumulates them in the graph state's
    ``messages`` list. Both render through ``infra.tracing.render_trace``.
    """

    index: int
    kind: TraceKind
    label: str = ""
    text: str = ""
    cache: CacheVerdict | None = None


@dataclass(frozen=True, slots=True)
class TriageRun:
    """Everything one run produced — the object ``test_parity.py`` compares across parts."""

    decision: TriageDecision
    trace: tuple[TraceStep, ...] = ()
    iterations: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``--json`` output."""
        return {
            "decision": self.decision.to_dict(),
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "trace": [
                {
                    "index": step.index,
                    "kind": step.kind,
                    "label": step.label,
                    "text": step.text,
                    "cache": step.cache,
                }
                for step in self.trace
            ],
        }


@dataclass
class Counter:
    """A mutable call counter used for cache stats and test spies."""

    value: int = 0

    def increment(self, by: int = 1) -> None:
        """Add ``by`` to the count."""
        self.value += by


@dataclass
class RunCounters:
    """Mutable per-run tallies the loop increments as it goes."""

    iterations: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    trace: list[TraceStep] = field(default_factory=list)
