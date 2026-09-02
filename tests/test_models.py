"""Domain model tests: enum round-trips, rendering, and serialisation."""

from __future__ import annotations

import json
from datetime import date

from triage_core.domain.models import (
    Action,
    Incident,
    Severity,
    StopReason,
    ToolResult,
    TraceStep,
    TriageDecision,
    TriageRun,
)


def test_enums_round_trip_through_their_values() -> None:
    assert Severity("critical") is Severity.CRITICAL
    assert Action("page_on_call") is Action.PAGE_ON_CALL
    assert StopReason("iteration_limit") is StopReason.ITERATION_LIMIT


def test_enums_render_as_bare_values_in_f_strings() -> None:
    """The trace must read ``action=file_ticket``, not ``action=Action.FILE_TICKET``."""
    assert f"{Action.FILE_TICKET}" == "file_ticket"
    assert f"{Severity.CRITICAL}" == "critical"
    assert f"{StopReason.FINAL_ANSWER}" == "final_answer"


def test_tool_result_renders_detail_on_success_and_error_on_failure() -> None:
    ok: ToolResult[str] = ToolResult(tool_name="t", ok=True, payload="p", detail="all good")
    assert ok.render() == "all good"

    bad: ToolResult[str] = ToolResult(tool_name="t", ok=False, error="it broke")
    assert bad.render() == "error: it broke"


def test_incident_serialises_dates_as_iso_strings() -> None:
    incident = Incident(
        incident_id="INC-1",
        signature="sig",
        severity=Severity.CRITICAL,
        opened_at=date(2026, 8, 2),
        resolved=True,
        resolution="fixed",
        occurrences=2,
    )
    payload = incident.to_dict()
    assert payload["opened_at"] == "2026-08-02"
    assert payload["severity"] == "critical"
    assert json.loads(json.dumps(payload))


def test_triage_decision_serialises_for_the_json_flag() -> None:
    decision = TriageDecision(
        action=Action.PAGE_ON_CALL,
        severity=Severity.CRITICAL,
        confidence=0.9,
        justification="novel failure",
        matched_incidents=("INC-1", "INC-2"),
    )
    payload = decision.to_dict()
    assert payload["action"] == "page_on_call"
    assert payload["matched_incidents"] == ["INC-1", "INC-2"]
    assert payload["complete"] is True
    assert json.loads(json.dumps(payload))


def test_triage_run_serialises_including_the_trace() -> None:
    run = TriageRun(
        decision=TriageDecision(
            action=Action.IGNORE,
            severity=Severity.INFO,
            confidence=0.95,
            justification="routine",
        ),
        trace=(TraceStep(index=1, kind="observation", text="obs", cache="hit"),),
        iterations=1,
        tool_calls=1,
        llm_calls=1,
        cache_hits=1,
    )
    payload = json.loads(json.dumps(run.to_dict()))
    assert payload["trace"][0]["cache"] == "hit"
    assert payload["cache_hits"] == 1


def test_decision_defaults_to_a_complete_final_answer() -> None:
    decision = TriageDecision(
        action=Action.IGNORE, severity=Severity.INFO, confidence=1.0, justification="j"
    )
    assert decision.complete
    assert decision.stop_reason is StopReason.FINAL_ANSWER
    assert decision.matched_incidents == ()
