"""The tool registry: a plain ``dict`` of name to callable tool.

Serves both parts. This *is* the dict Rule R2 requires Part 1's loop to dispatch through, and
Part 2's ``act`` node dispatches through the same object — one tool implementation, two
orchestrations (Rule R3).

Built per run rather than as a module-level global, so tests get fresh instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from triage_core.domain.ports import TextCompleter
from triage_core.domain.prompts import build_react_system
from triage_core.tools.base import Tool
from triage_core.tools.classify_severity import ClassifySeverityTool
from triage_core.tools.incident_lookup import IncidentLookupTool
from triage_core.tools.recommend_action import RecommendActionTool


def build_registry(
    *,
    llm: TextCompleter,
    incidents_path: Path,
    lookback_days: int = 90,
    today: date | None = None,
) -> dict[str, Tool]:
    """Construct the three tools and index them by name.

    Args:
        llm: Injected completer, used only by ``classify_severity``'s fallback path.
        incidents_path: Location of ``incidents.json``.
        lookback_days: Recurrence window for ``lookup_incidents``.
        today: Injectable "now" so tests are not time-dependent.

    Returns:
        A dict mapping tool name to tool, in the order the model should see them.

    Raises:
        DataError: The incident history is missing or corrupt.
    """
    tools: list[Tool] = [
        ClassifySeverityTool(llm=llm),
        IncidentLookupTool(incidents_path, lookback_days=lookback_days, today=today),
        RecommendActionTool(),
    ]
    return {tool.name: tool for tool in tools}


def describe_tools(registry: Mapping[str, Tool]) -> str:
    """Render a registry into the ReAct system prompt.

    Args:
        registry: The tools available this run.

    Returns:
        The fully rendered system prompt (prompt text itself lives in ``domain/prompts.py``).
    """
    return build_react_system(registry.values())
