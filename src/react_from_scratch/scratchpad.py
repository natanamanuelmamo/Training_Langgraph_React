"""The accumulated Thought / Action / Observation transcript (``task01.md`` §3).

Part 1 only. This is the agent's memory: a stateless LLM call behaves like an agent purely
because this object grows every iteration and is re-rendered into the next prompt.

Rendering is delegated to ``triage_core.infra.tracing.render_transcript`` so that Part 2's
``reason`` node produces byte-identical prompts from its own state — which is what makes the
parity test meaningful rather than coincidental.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from triage_core.domain.models import TraceStep
from triage_core.domain.prompts import SCRATCHPAD_HEADER
from triage_core.infra.tracing import render_transcript


class Scratchpad:
    """A growing ReAct transcript for one log line."""

    def __init__(self, log_line: str) -> None:
        """Start a transcript.

        Args:
            log_line: The line under triage.
        """
        self.log_line = log_line
        self._steps: list[TraceStep] = []
        self._iteration = 0

    @property
    def steps(self) -> tuple[TraceStep, ...]:
        """Every step recorded so far, in order."""
        return tuple(self._steps)

    @property
    def observations(self) -> tuple[str, ...]:
        """Just the Observation texts, oldest first."""
        return tuple(step.text for step in self._steps if step.kind == "observation")

    def begin_iteration(self, iteration: int) -> None:
        """Record which iteration subsequent steps belong to.

        Args:
            iteration: The 1-based iteration number.
        """
        self._iteration = iteration

    def add_thought(self, text: str) -> TraceStep:
        """Append a Thought.

        Args:
            text: The model's reasoning for this turn.

        Returns:
            The recorded step, so the caller can also hand it to a tracer.
        """
        return self._append(TraceStep(index=self._iteration, kind="thought", text=text))

    def add_action(self, name: str, args: Mapping[str, Any]) -> TraceStep:
        """Append an Action and its input.

        Args:
            name: The tool name the model chose.
            args: The parsed ``Action Input`` object.

        Returns:
            The recorded step.
        """
        return self._append(
            TraceStep(
                index=self._iteration,
                kind="action",
                label=name,
                text=json.dumps(dict(args), sort_keys=True),
            )
        )

    def add_observation(self, text: str, *, cache: str | None = None) -> TraceStep:
        """Append an Observation.

        Args:
            text: What the tool returned, or the error the loop turned into an Observation.
            cache: ``"hit"`` or ``"miss"`` when the call went through the cache.

        Returns:
            The recorded step.
        """
        verdict = cache if cache in {"hit", "miss"} else None
        return self._append(
            TraceStep(
                index=self._iteration,
                kind="observation",
                text=text,
                cache=verdict,  # type: ignore[arg-type]
            )
        )

    def add_final(self, text: str) -> TraceStep:
        """Append the concluding Final Answer line.

        Args:
            text: The chosen action, for the display trace.

        Returns:
            The recorded step.
        """
        return self._append(TraceStep(index=self._iteration, kind="final", text=text))

    def render(self) -> str:
        """Render the transcript for the next reason step.

        Returns:
            The header plus every Thought / Action / Observation so far.
        """
        return SCRATCHPAD_HEADER + "\n" + render_transcript(self.log_line, self._steps)

    def _append(self, step: TraceStep) -> TraceStep:
        """Store a step and hand it back."""
        self._steps.append(step)
        return step
