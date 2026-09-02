"""Step 2 of the triage chain: find past incidents matching a log's signature.

Serves both parts (``task01.md`` §2). This is the step that depends on step 1's output — the
assignment's required "step that depends on looking something up" — and the deterministic,
repeated call that Part 2 caches (``task02.md`` §4).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar

from triage_core.domain.errors import DataError, ToolInputError
from triage_core.domain.models import Incident, IncidentMatches, Severity, ToolResult

_REQUIRED_FIELDS = frozenset(
    {"incident_id", "signature", "severity", "opened_at", "resolved", "resolution", "occurrences"}
)


def load_incidents(path: Path) -> tuple[Incident, ...]:
    """Read and validate the seed incident history.

    Args:
        path: Location of ``incidents.json``.

    Returns:
        Every incident in the file, in declaration order.

    Raises:
        DataError: The file is missing, is not valid JSON, or has the wrong shape. This is a
            setup problem rather than an agent problem, so it is fatal rather than an
            Observation.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataError(f"incident history not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"incident history at {path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise DataError(
            f"incident history at {path} must be a JSON array, got {type(raw).__name__}"
        )

    incidents: list[Incident] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DataError(f"incident #{position} in {path} is not an object")
        if missing := _REQUIRED_FIELDS - set(item):
            raise DataError(
                f"incident #{position} in {path} is missing field(s): {sorted(missing)}"
            )
        try:
            resolution = item["resolution"]
            incidents.append(
                Incident(
                    incident_id=str(item["incident_id"]),
                    signature=str(item["signature"]),
                    severity=Severity(str(item["severity"]).lower()),
                    opened_at=date.fromisoformat(str(item["opened_at"])),
                    resolved=bool(item["resolved"]),
                    resolution=None if resolution is None else str(resolution),
                    occurrences=int(item["occurrences"]),
                )
            )
        except (ValueError, TypeError) as exc:
            raise DataError(f"incident #{position} in {path} has a bad field value: {exc}") from exc
    return tuple(incidents)


class IncidentLookupTool:
    """Search past incidents for ones sharing a log line's signature."""

    name: ClassVar[str] = "lookup_incidents"
    description: ClassVar[str] = (
        "Search the incident history for past incidents with the same signature. Returns each "
        "match with its resolution, recurrence count and date. Call this after you know the "
        "severity and signature — the escalation policy depends on whether this failure is "
        "novel, known-and-fixed, or recurring. The Observation ends with an "
        "'incidents = [...]' line you can pass straight to recommend_action."
    )
    input_schema: ClassVar[dict[str, str]] = {
        "signature": "string — the signature returned by classify_severity",
        "severity": "string — one of critical, warning, info",
    }

    def __init__(
        self,
        incidents_path: Path,
        lookback_days: int = 90,
        *,
        today: date | None = None,
    ) -> None:
        """Load the incident history once for this registry instance.

        Args:
            incidents_path: Location of ``incidents.json``.
            lookback_days: Window for the recurrence count the escalation policy reads.
            today: Injectable "now" so tests are not time-dependent.

        Raises:
            DataError: The seed data is missing or corrupt.
        """
        self._incidents = load_incidents(incidents_path)
        self._lookback_days = lookback_days
        self._today = today or date.today()

    def run(self, **kwargs: Any) -> ToolResult[IncidentMatches]:
        """Look up incidents matching a signature.

        Zero matches is a **success**, not an error — "novel failure" is a real answer and it
        drives the ``page_on_call`` branch of the escalation policy.

        Args:
            **kwargs: Must contain ``signature`` (string) and ``severity`` (a valid severity).

        Returns:
            A result carrying the matches, their total recurrence count inside the lookback
            window, and the most recent occurrence.

        Raises:
            ToolInputError: An argument is missing, of the wrong type, or not a valid severity.
        """
        signature = kwargs.get("signature")
        severity_raw = kwargs.get("severity")

        if signature is None:
            raise ToolInputError("lookup_incidents requires a 'signature' argument.")
        if not isinstance(signature, str) or not signature.strip():
            raise ToolInputError("'signature' must be a non-empty string.")
        if severity_raw is None:
            raise ToolInputError("lookup_incidents requires a 'severity' argument.")
        if not isinstance(severity_raw, str):
            raise ToolInputError(f"'severity' must be a string, got {type(severity_raw).__name__}.")
        try:
            Severity(severity_raw.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"'severity' must be one of critical, warning, info — got {severity_raw!r}."
            ) from exc
        if unexpected := set(kwargs) - {"signature", "severity"}:
            raise ToolInputError(
                f"lookup_incidents got unexpected argument(s): {sorted(unexpected)}. "
                "It takes only 'signature' and 'severity'."
            )

        cutoff = self._today - timedelta(days=self._lookback_days)
        matches = tuple(
            sorted(
                (
                    incident
                    for incident in self._incidents
                    if incident.signature == signature.strip() and incident.opened_at >= cutoff
                ),
                key=lambda incident: incident.opened_at,
                reverse=True,
            )
        )
        result = IncidentMatches(
            signature=signature.strip(),
            matches=matches,
            total_occurrences=sum(incident.occurrences for incident in matches),
            lookback_days=self._lookback_days,
            latest=matches[0].opened_at if matches else None,
        )
        return ToolResult(
            tool_name=self.name,
            ok=True,
            payload=result,
            detail=self._render(result),
        )

    @staticmethod
    def _render(result: IncidentMatches) -> str:
        """Render the Observation text for a set of matches.

        The Observation ends with a machine-readable ``incidents = [...]`` line. The next step
        in the chain, ``recommend_action``, needs these objects, and the only channel between
        the two is the JSON the model writes in its ``Action Input``. Handing it a payload it
        can copy verbatim removes a transcription step that models get wrong.
        """
        payload = json.dumps(
            [
                {
                    "incident_id": incident.incident_id,
                    "resolved": incident.resolved,
                    "resolution": incident.resolution,
                    "occurrences": incident.occurrences,
                }
                for incident in result.matches
            ]
        )
        if not result.matches:
            return (
                f"0 matches for {result.signature} in the last {result.lookback_days} days "
                f"— novel failure\nincidents = {payload}"
            )
        parts = [
            f"{incident.incident_id} ({incident.opened_at.isoformat()}, "
            + (
                f"resolved: {incident.resolution}"
                if incident.resolved and incident.resolution
                else "unresolved"
            )
            + f", x{incident.occurrences})"
            for incident in result.matches
        ]
        return (
            f"{len(result.matches)} matches, {result.total_occurrences} total occurrences "
            f"in the last {result.lookback_days} days — "
            + "; ".join(parts)
            + f"\nincidents = {payload}"
        )
