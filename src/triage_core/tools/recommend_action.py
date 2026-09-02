"""Step 3 of the triage chain: apply the escalation policy.

Serves both parts (``task01.md`` §2). The policy lives here in code, never in a prompt: the
LLM decides *which tool to call next*, this function decides *what the escalation rule is*.
That separation is what makes the agent auditable.

Rules are evaluated in an explicit precedence order (see :data:`POLICY_ORDER`). Two points
where ``task01.md`` §2's table is ambiguous are resolved here and documented in
``docs/task_1_implementation.md`` §11:

1. A critical failure recurring three or more times outranks a known resolution — recurring
   *despite* a known fix is precisely a regression, and regressions page.
2. A critical failure with prior incidents that were never resolved is not in the spec's
   table at all. It pages: it is critical, it has happened before, and nobody fixed it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from triage_core.domain.errors import ToolInputError
from triage_core.domain.models import (
    Action,
    ActionRecommendation,
    Incident,
    Severity,
    ToolResult,
)
from triage_core.tools.incident_coercion import coerce_incidents

#: Recurrences inside the lookback window at or above which a critical failure is a regression.
REGRESSION_THRESHOLD = 3

#: The escalation table, in evaluation order. Documented for the README requirement map (R13).
POLICY_ORDER: tuple[str, ...] = (
    "info_ignore",
    "warning_ticket",
    "critical_novel",
    "critical_regression",
    "critical_known_fix",
    "critical_unresolved",
)


def decide_action(
    severity: Severity, incidents: Sequence[Incident], total_occurrences: int
) -> ActionRecommendation:
    """Apply the escalation policy.

    This is the pure heart of the tool, separated so both the tool and the tests can call it
    without constructing a ``ToolResult``.

    Args:
        severity: The classified severity.
        incidents: Matching past incidents inside the lookback window.
        total_occurrences: Their combined recurrence count.

    Returns:
        The action, the rule that produced it, and a justification a human can act on.
    """
    if severity is Severity.INFO:
        return ActionRecommendation(
            action=Action.IGNORE,
            justification="Informational log line; no action required.",
            policy_rule="info_ignore",
            confidence=0.95,
        )

    if severity is Severity.WARNING:
        return ActionRecommendation(
            action=Action.FILE_TICKET,
            justification=(
                "Degraded behaviour rather than an outage; track it in a ticket instead of paging."
            ),
            policy_rule="warning_ticket",
            confidence=0.85,
        )

    # severity is CRITICAL from here down.
    if not incidents:
        return ActionRecommendation(
            action=Action.PAGE_ON_CALL,
            justification=(
                "Critical failure with no matching prior incidents — novel, so nobody has a "
                "known fix. Page on-call."
            ),
            policy_rule="critical_novel",
            confidence=0.9,
        )

    if total_occurrences >= REGRESSION_THRESHOLD:
        return ActionRecommendation(
            action=Action.PAGE_ON_CALL,
            justification=(
                f"Critical failure recurring {total_occurrences} times in the lookback window "
                f"({_ids(incidents)}) — treated as a regression. Page on-call."
            ),
            policy_rule="critical_regression",
            confidence=0.9,
        )

    if resolved := [i for i in incidents if i.resolved and i.resolution]:
        prior = resolved[0]
        return ActionRecommendation(
            action=Action.FILE_TICKET,
            justification=(
                f"Recurring known failure; prior resolution exists ({prior.incident_id}: "
                f"{prior.resolution}). File a ticket referencing it rather than paging."
            ),
            policy_rule="critical_known_fix",
            confidence=0.86,
        )

    return ActionRecommendation(
        action=Action.PAGE_ON_CALL,
        justification=(
            f"Critical failure seen before ({_ids(incidents)}) but never resolved. Page on-call."
        ),
        policy_rule="critical_unresolved",
        confidence=0.88,
    )


def _ids(incidents: Sequence[Incident]) -> str:
    """Render incident ids for a justification string."""
    return ", ".join(incident.incident_id for incident in incidents)


class RecommendActionTool:
    """Turn a severity plus incident history into the final escalation decision."""

    name: ClassVar[str] = "recommend_action"
    description: ClassVar[str] = (
        "Apply the escalation policy to a severity and the incidents found for its signature, "
        "producing the final action (page_on_call / file_ticket / ignore) and a justification. "
        "Call this once you have both the severity and the incident lookup result — do not "
        "decide the action yourself."
    )
    input_schema: ClassVar[dict[str, str]] = {
        "severity": "string — one of critical, warning, info",
        "incidents": "list — the matches returned by lookup_incidents, possibly empty",
    }

    def run(self, **kwargs: Any) -> ToolResult[ActionRecommendation]:
        """Recommend an action.

        Args:
            **kwargs: Must contain ``severity``; ``incidents`` defaults to empty.

        Returns:
            A result carrying the action, the policy rule that fired, and the justification.

        Raises:
            ToolInputError: ``severity`` is missing or invalid, or an incident is malformed.
        """
        severity_raw = kwargs.get("severity")
        if severity_raw is None:
            raise ToolInputError("recommend_action requires a 'severity' argument.")
        if not isinstance(severity_raw, str):
            raise ToolInputError(f"'severity' must be a string, got {type(severity_raw).__name__}.")
        try:
            severity = Severity(severity_raw.lower())
        except ValueError as exc:
            raise ToolInputError(
                f"'severity' must be one of critical, warning, info — got {severity_raw!r}."
            ) from exc
        if unexpected := set(kwargs) - {"severity", "incidents"}:
            raise ToolInputError(
                f"recommend_action got unexpected argument(s): {sorted(unexpected)}. "
                "It takes only 'severity' and 'incidents'."
            )

        incidents, total = coerce_incidents(kwargs.get("incidents"))
        recommendation = decide_action(severity, incidents, total)
        return ToolResult(
            tool_name=self.name,
            ok=True,
            payload=recommendation,
            detail=(
                f"action={recommendation.action} (rule={recommendation.policy_rule}) "
                f"— {recommendation.justification}"
            ),
        )
