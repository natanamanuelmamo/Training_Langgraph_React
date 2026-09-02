"""The tool contract shared by both parts of the assignment (``task01.md`` §2).

Serves both parts. Part 1's ``while`` loop and Part 2's ``act`` node dispatch through the same
registry of objects satisfying this protocol — that is what Rule R3 means in practice.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from triage_core.domain.models import ToolResult


@runtime_checkable
class Tool(Protocol):
    """A callable capability the agent can invoke by name.

    Implementations return a structured :class:`ToolResult`, never a raw string. They raise
    :class:`~triage_core.domain.errors.ToolInputError` for structurally wrong arguments and
    return ``ToolResult(ok=False, ...)`` when they ran but could not produce an answer. Both
    outcomes become Observations; the split keeps "found nothing" distinguishable from
    "called wrong".
    """

    @property
    def name(self) -> str:
        """The exact string the model writes after ``Action:``."""
        ...

    @property
    def description(self) -> str:
        """What the model reads when deciding whether to call this tool."""
        ...

    @property
    def input_schema(self) -> dict[str, str]:
        """Argument name to a human-readable description of its type and meaning."""
        ...

    def run(self, **kwargs: Any) -> ToolResult[Any]:
        """Execute the tool.

        Args:
            **kwargs: The parsed ``Action Input`` object.

        Returns:
            A structured result the agent renders into an Observation.

        Raises:
            ToolInputError: The arguments were structurally wrong.
            ToolExecutionError: The tool failed for a reason the caller could not prevent.
        """
        ...
