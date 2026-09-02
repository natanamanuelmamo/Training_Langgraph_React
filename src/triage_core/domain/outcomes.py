"""Build a :class:`TriageDecision` from what a run produced.

Serves both parts. Part 1's loop and Part 2's ``decide`` / ``halt`` nodes call the same two
functions, so an identical run yields an identical decision — which is what makes
``tests/test_parity.py`` a real check rather than a coincidence (Rule R3).

Pure: imports only the stdlib and sibling ``domain`` modules (Rule R6).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from triage_core.domain.models import (
    Action,
    Severity,
    StopReason,
    TriageDecision,
)


def decision_from_payload(payload: Mapping[str, Any]) -> TriageDecision:
    """Convert a validated ``Final Answer`` object into a decision.

    The parser has already checked the payload's shape, so this only converts.

    Args:
        payload: The parsed ``Final Answer`` object.

    Returns:
        The completed decision.

    Raises:
        ValueError: A field could not be converted. Callers turn this into an Observation
            rather than letting it escape — a bad final answer is a correctable mistake.
        KeyError: A required field was absent.
        TypeError: A field held an unusable type.
    """
    return TriageDecision(
        action=Action(str(payload["action"]).lower()),
        severity=Severity(str(payload["severity"]).lower()),
        confidence=float(payload["confidence"]),
        justification=str(payload["justification"]).strip(),
        matched_incidents=tuple(str(item) for item in payload["matched_incidents"]),
        complete=True,
        stop_reason=StopReason.FINAL_ANSWER,
    )


def severity_from_observations(observations: Iterable[str]) -> Severity:
    """Recover the last severity the run actually observed.

    Args:
        observations: Observation texts, oldest first.

    Returns:
        The most recent severity seen, defaulting to ``warning`` when none was observed —
        the safe default, since it routes to a human without waking anyone.
    """
    severity = Severity.WARNING
    for observation in observations:
        if "severity=" in observation:
            candidate = observation.split("severity=", 1)[1].split()[0]
            try:
                severity = Severity(candidate.lower())
            except ValueError:
                continue
    return severity


def partial_decision(observations: Iterable[str], max_iterations: int) -> TriageDecision:
    """Build the result for a run that hit the iteration cap.

    Never raises: both parts must always hand back a well-formed result rather than spinning
    or crashing (Rule R10).

    Args:
        observations: Observation texts from the run, oldest first.
        max_iterations: The cap that was hit, quoted in the justification.

    Returns:
        A partial decision with ``complete=False``.
    """
    return TriageDecision(
        action=Action.FILE_TICKET,
        severity=severity_from_observations(observations),
        confidence=0.0,
        justification=(
            f"Stopped after the {max_iterations}-iteration limit without a final answer. "
            "Filing a ticket so a human reviews it — this is a partial result, not a verdict."
        ),
        matched_incidents=(),
        complete=False,
        stop_reason=StopReason.ITERATION_LIMIT,
    )


def rejected_decision(decision: TriageDecision, note: str) -> TriageDecision:
    """Rewrite a decision after a human declined the sensitive action.

    Part 2 only, but it lives here so the decision-shaping rules stay in one place. Rejection
    must change the *outcome*, not merely the path — a checkpoint that delays the same result
    is not a checkpoint (``task02.md`` §5.5).

    Args:
        decision: The decision the graph proposed.
        note: The reviewer's reason, folded into the justification.

    Returns:
        A ticket instead of a page, marked as ended by human rejection.
    """
    reason = note.strip() or "no reason given"
    return TriageDecision(
        action=Action.FILE_TICKET,
        severity=decision.severity,
        confidence=decision.confidence,
        justification=(
            f"Paging was rejected by a human reviewer ({reason}). "
            f"Filing a ticket instead. Original recommendation: {decision.justification}"
        ),
        matched_incidents=decision.matched_incidents,
        complete=True,
        stop_reason=StopReason.HUMAN_REJECTED,
    )
