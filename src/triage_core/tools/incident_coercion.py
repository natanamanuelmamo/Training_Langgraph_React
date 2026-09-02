"""Adapt whatever the model passed as ``incidents`` into typed :class:`Incident` objects.

Serves both parts, via ``recommend_action``. Split out so ``recommend_action.py`` contains
only the escalation policy — that file is what a reviewer opens to confirm the policy lives in
code rather than in a prompt, and adapter noise does not belong in it (Rule R11).

``Action Input`` reaches the tool as parsed JSON, so ``incidents`` is normally a list of plain
dicts transcribed from the previous Observation. A caller holding the typed
:class:`IncidentMatches` may pass that instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from triage_core.domain.errors import ToolInputError
from triage_core.domain.models import Incident, IncidentMatches, Severity

#: Stand-in for an incident whose ``opened_at`` the model did not transcribe. The lookback
#: filter already ran inside ``lookup_incidents``, so the policy never reads this date — it
#: exists only to keep :class:`Incident` fully populated.
_UNKNOWN_DATE = date(1970, 1, 1)


def coerce_incidents(value: Any) -> tuple[list[Incident], int]:
    """Normalise the ``incidents`` argument into incidents plus a total occurrence count.

    Args:
        value: ``None``, an :class:`IncidentMatches`, or a sequence of incident dicts.

    Returns:
        The incidents and the sum of their occurrence counts.

    Raises:
        ToolInputError: The value is not a recognised shape, or an entry is malformed.
    """
    if isinstance(value, IncidentMatches):
        return list(value.matches), value.total_occurrences

    if value is None:
        return [], 0

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ToolInputError(
            "'incidents' must be a list of incident objects (or an empty list), got "
            f"{type(value).__name__}."
        )

    incidents: list[Incident] = []
    total = 0
    for position, item in enumerate(value):
        incident = _coerce_one(item, position)
        incidents.append(incident)
        total += incident.occurrences
    return incidents, total


def _coerce_one(item: Any, position: int) -> Incident:
    """Convert a single entry, which may already be typed.

    Raises:
        ToolInputError: The entry is not an object, or a field is unusable.
    """
    if isinstance(item, Incident):
        return item

    if not isinstance(item, dict):
        raise ToolInputError(f"incident #{position} must be an object, got {type(item).__name__}.")

    incident_id = item.get("incident_id")
    if not isinstance(incident_id, str) or not incident_id:
        raise ToolInputError(f"incident #{position} is missing a string 'incident_id'.")

    try:
        occurrences = int(item.get("occurrences", 1))
    except (TypeError, ValueError) as exc:
        raise ToolInputError(
            f"incident #{position} has a non-numeric 'occurrences' value."
        ) from exc

    try:
        severity = Severity(str(item.get("severity", "critical")).lower())
    except ValueError as exc:
        raise ToolInputError(
            f"incident #{position} has an unrecognised 'severity': {item.get('severity')!r}."
        ) from exc

    resolution = item.get("resolution")
    return Incident(
        incident_id=incident_id,
        signature=str(item.get("signature", "")),
        severity=severity,
        opened_at=_coerce_date(item.get("opened_at"), position),
        resolved=bool(item.get("resolved", False)),
        resolution=None if resolution is None else str(resolution),
        occurrences=occurrences,
    )


def _coerce_date(value: Any, position: int) -> date:
    """Parse an ISO date, tolerating the model omitting it entirely.

    Raises:
        ToolInputError: A value was supplied but could not be parsed.
    """
    if isinstance(value, date):
        return value
    if value is None:
        return _UNKNOWN_DATE
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ToolInputError(
            f"incident #{position} has an unparseable 'opened_at' value: {value!r}."
        ) from exc
