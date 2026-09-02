"""Protocols that let lower layers use higher-layer capabilities without importing them.

Serves both parts. ``classify_severity`` needs an LLM fallback for log lines its rules cannot
match (``task01.md`` §2). If ``tools/`` imported ``llm/``, the layering in Rule R6
(``tools`` sits *below* ``llm``) would invert. Instead the tool depends on the
:class:`TextCompleter` protocol declared here and receives the concrete client by injection.

Protocols are structural, so ``llm/client.py`` satisfies this without importing it — no import
edge is created in either direction.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextCompleter(Protocol):
    """Anything that can turn a (system, user) prompt pair into text.

    Satisfied by ``AnthropicLLM``, ``ScriptedLLM`` and ``FakeLLM`` in ``triage_core.llm.client``.
    """

    def complete(self, system: str, user: str) -> str:
        """Return the model's text response for the given prompts."""
        ...


@runtime_checkable
class ToolSpec(Protocol):
    """The part of a tool that the prompt layer needs to describe it to the model.

    ``domain`` sits below ``tools``, so ``prompts.render_tool_block`` cannot import the real
    ``Tool`` protocol. It depends on this narrower view instead; ``tools.base.Tool`` satisfies
    it structurally.
    """

    @property
    def name(self) -> str:
        """The exact string the model writes after ``Action:``."""
        ...

    @property
    def description(self) -> str:
        """One or two sentences telling the model when to reach for this tool."""
        ...

    @property
    def input_schema(self) -> dict[str, str]:
        """Argument name to a human-readable description of its type and meaning."""
        ...
