"""Typed exceptions for the Log-Triage Agent.

Serves both parts. The distinction that matters: ``ToolInputError``, ``ToolExecutionError``
and ``ParseError`` are all caught by the agent loop and turned into Observations (Rule R10),
so the model can correct itself. ``DataError`` is the one that may end the process — it means
the seed data is broken, which is a setup problem, not an agent problem.
"""

from __future__ import annotations


class TriageError(Exception):
    """Base class for every error this package raises."""


class ToolInputError(TriageError):
    """A tool was called with structurally wrong arguments.

    Becomes an Observation so the model can retry with a corrected ``Action Input``.
    """


class ToolExecutionError(TriageError):
    """A tool failed while running, for a reason the caller could not have prevented."""


class ParseError(TriageError):
    """The LLM's output did not match the required ReAct grammar.

    The message is fed back to the model verbatim as an Observation, so it must say exactly
    what was wrong and what was expected.
    """


class DataError(TriageError):
    """Seed data is missing or corrupt. Fatal — this is a setup problem, not an agent problem."""
