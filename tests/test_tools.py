"""Tool tests: happy path, bad input and an edge case for each (``task01.md`` §7)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from tests.conftest import CERT_LINE, INFO_LINE, OOM_LINE, POOL_LINE, TODAY, WARN_LINE

from triage_core.domain.errors import DataError, ToolInputError
from triage_core.domain.models import (
    Action,
    ActionRecommendation,
    Incident,
    IncidentMatches,
    Severity,
)
from triage_core.tools.base import Tool
from triage_core.tools.classify_severity import ClassifySeverityTool
from triage_core.tools.incident_lookup import load_incidents
from triage_core.tools.recommend_action import decide_action
from triage_core.tools.registry import build_registry, describe_tools

# ---------------------------------------------------------------------------
# classify_severity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "severity", "signature", "rule"),
    [
        (POOL_LINE, Severity.CRITICAL, "db.pool.exhausted:orders-primary", "db_pool_exhausted"),
        (OOM_LINE, Severity.CRITICAL, "oom.killed:recommendation-worker", "oom_killed"),
        (CERT_LINE, Severity.CRITICAL, "tls.cert.expired:checkout-gateway", "tls_cert_expired"),
        (WARN_LINE, Severity.WARNING, "net.timeout:search-api", "upstream_timeout"),
        (INFO_LINE, Severity.INFO, "cache.warm:catalog-api", "cache_warm"),
    ],
)
def test_classify_severity_happy_path(
    registry: dict[str, Tool], line: str, severity: Severity, signature: str, rule: str
) -> None:
    result = registry["classify_severity"].run(log_line=line)
    assert result.ok
    assert result.payload is not None
    assert result.payload.severity is severity
    assert result.payload.signature == signature
    assert result.payload.matched_rule == rule


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"log_line": 42},
        {"log_line": "   "},
        {"log_line": POOL_LINE, "extra": "nope"},
    ],
)
def test_classify_severity_bad_input(registry: dict[str, Tool], kwargs: dict[str, object]) -> None:
    with pytest.raises(ToolInputError):
        registry["classify_severity"].run(**kwargs)


def test_classify_severity_falls_back_to_the_llm_when_no_rule_matches() -> None:
    """Edge case: an unrecognised line reaches the injected completer."""

    class _Stub:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            return "severity: critical\nsignature: queue.stalled:billing-worker"

    stub = _Stub()
    tool = ClassifySeverityTool(llm=stub)  # type: ignore[arg-type]
    result = tool.run(log_line="2026-08-30T00:00:00Z ERROR billing-worker  Something inexplicable")

    assert stub.calls == 1
    assert result.ok
    assert result.payload is not None
    assert result.payload.matched_rule == "llm_fallback"
    assert result.payload.signature == "queue.stalled:billing-worker"
    assert result.payload.confidence <= 0.6


def test_classify_severity_reports_an_unusable_fallback_answer() -> None:
    """Edge case: a garbage fallback is an ``ok=False`` result, not a guess."""

    class _Garbage:
        def complete(self, system: str, user: str) -> str:
            return "I am not sure, sorry!"

    tool = ClassifySeverityTool(llm=_Garbage())  # type: ignore[arg-type]
    result = tool.run(log_line="2026-08-30T00:00:00Z ERROR billing-worker  Something inexplicable")

    assert not result.ok
    assert result.payload is None
    assert "unusable" in (result.error or "")


# ---------------------------------------------------------------------------
# lookup_incidents
# ---------------------------------------------------------------------------


def test_lookup_incidents_happy_path(registry: dict[str, Tool]) -> None:
    result = registry["lookup_incidents"].run(
        signature="db.pool.exhausted:orders-primary", severity="critical"
    )
    assert result.ok
    assert result.payload is not None
    assert [i.incident_id for i in result.payload.matches] == ["INC-0002", "INC-0001"]
    assert result.payload.total_occurrences == 2
    assert result.payload.latest == date(2026, 8, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"severity": "critical"},
        {"signature": "db.pool.exhausted:orders-primary"},
        {"signature": "", "severity": "critical"},
        {"signature": "x", "severity": "catastrophic"},
        {"signature": "x", "severity": 3},
        {"signature": "x", "severity": "critical", "extra": 1},
    ],
)
def test_lookup_incidents_bad_input(registry: dict[str, Tool], kwargs: dict[str, object]) -> None:
    with pytest.raises(ToolInputError):
        registry["lookup_incidents"].run(**kwargs)


def test_lookup_incidents_treats_no_matches_as_success(registry: dict[str, Tool]) -> None:
    """Edge case: "novel failure" is a real answer, and it drives the paging branch."""
    result = registry["lookup_incidents"].run(
        signature="tls.cert.expired:checkout-gateway", severity="critical"
    )
    assert result.ok
    assert result.payload is not None
    assert result.payload.matches == ()
    assert "novel failure" in result.render()


def test_lookup_incidents_excludes_records_outside_the_lookback(
    registry: dict[str, Tool],
) -> None:
    """Edge case: INC-0009 is in the file but 230 days old, so it must not appear."""
    result = registry["lookup_incidents"].run(
        signature="db.pool.exhausted:orders-primary", severity="critical"
    )
    assert result.payload is not None
    assert "INC-0009" not in [i.incident_id for i in result.payload.matches]
    assert result.payload.total_occurrences == 2


def test_lookup_incidents_observation_carries_a_forwardable_payload(
    registry: dict[str, Tool],
) -> None:
    """The Observation ends with the JSON ``recommend_action`` needs next."""
    rendered = (
        registry["lookup_incidents"]
        .run(signature="db.pool.exhausted:orders-primary", severity="critical")
        .render()
    )
    assert "incidents = [" in rendered


def test_load_incidents_rejects_corrupt_data(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('[{"incident_id": "INC-1"}]', encoding="utf-8")
    with pytest.raises(DataError):
        load_incidents(bad)

    missing = tmp_path / "nope.json"
    with pytest.raises(DataError):
        load_incidents(missing)

    not_json = tmp_path / "x.json"
    not_json.write_text("{{{", encoding="utf-8")
    with pytest.raises(DataError):
        load_incidents(not_json)


# ---------------------------------------------------------------------------
# recommend_action — every branch of the escalation policy
# ---------------------------------------------------------------------------


def _incident(
    incident_id: str = "INC-1",
    *,
    resolved: bool = True,
    resolution: str | None = "fixed",
    n: int = 1,
) -> Incident:
    return Incident(
        incident_id=incident_id,
        signature="sig",
        severity=Severity.CRITICAL,
        opened_at=date(2026, 8, 1),
        resolved=resolved,
        resolution=resolution,
        occurrences=n,
    )


@pytest.mark.parametrize(
    ("severity", "incidents", "total", "action", "rule"),
    [
        (Severity.INFO, [], 0, Action.IGNORE, "info_ignore"),
        (Severity.WARNING, [], 0, Action.FILE_TICKET, "warning_ticket"),
        (Severity.CRITICAL, [], 0, Action.PAGE_ON_CALL, "critical_novel"),
        (
            Severity.CRITICAL,
            [_incident(n=3)],
            3,
            Action.PAGE_ON_CALL,
            "critical_regression",
        ),
        (
            Severity.CRITICAL,
            [_incident("INC-1"), _incident("INC-2")],
            2,
            Action.FILE_TICKET,
            "critical_known_fix",
        ),
        (
            Severity.CRITICAL,
            [_incident(resolved=False, resolution=None)],
            1,
            Action.PAGE_ON_CALL,
            "critical_unresolved",
        ),
    ],
)
def test_every_policy_branch_is_reachable(
    severity: Severity,
    incidents: list[Incident],
    total: int,
    action: Action,
    rule: str,
) -> None:
    recommendation = decide_action(severity, incidents, total)
    assert recommendation.action is action
    assert recommendation.policy_rule == rule


def test_regression_outranks_a_known_fix() -> None:
    """Edge case: recurring *despite* a known fix is a regression, so it pages.

    Resolves the ordering ambiguity in ``task01.md`` §2 — see
    ``docs/task_1_implementation.md`` §11.
    """
    incidents = [_incident("INC-1", n=2), _incident("INC-2", n=2)]
    recommendation = decide_action(Severity.CRITICAL, incidents, 4)
    assert recommendation.action is Action.PAGE_ON_CALL
    assert recommendation.policy_rule == "critical_regression"


def test_recommend_action_accepts_the_json_round_trip(registry: dict[str, Tool]) -> None:
    """``Action Input`` arrives as parsed JSON, so a list of dicts must work."""
    result = registry["recommend_action"].run(
        severity="critical",
        incidents=[
            {"incident_id": "INC-1", "resolved": True, "resolution": "fixed", "occurrences": 1}
        ],
    )
    assert result.ok
    assert isinstance(result.payload, ActionRecommendation)
    assert result.payload.action is Action.FILE_TICKET


def test_recommend_action_accepts_the_typed_object(registry: dict[str, Tool]) -> None:
    matches = IncidentMatches(
        signature="sig",
        matches=(_incident(),),
        total_occurrences=1,
        lookback_days=90,
        latest=date(2026, 8, 1),
    )
    result = registry["recommend_action"].run(severity="critical", incidents=matches)
    assert result.ok


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"severity": "catastrophic"},
        {"severity": 7},
        {"severity": "critical", "incidents": "not-a-list"},
        {"severity": "critical", "incidents": [{"resolved": True}]},
        {"severity": "critical", "incidents": [{"incident_id": "X", "occurrences": "many"}]},
        {"severity": "critical", "extra": 1},
    ],
)
def test_recommend_action_bad_input(registry: dict[str, Tool], kwargs: dict[str, object]) -> None:
    with pytest.raises(ToolInputError):
        registry["recommend_action"].run(**kwargs)


def test_recommend_action_treats_missing_incidents_as_novel(registry: dict[str, Tool]) -> None:
    """Edge case: omitting ``incidents`` entirely means "none found"."""
    result = registry["recommend_action"].run(severity="critical")
    assert result.payload is not None
    assert result.payload.policy_rule == "critical_novel"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_is_a_plain_dict_of_tools(registry: dict[str, Tool]) -> None:
    """Rule R2: the loop dispatches through a real dict, not a framework."""
    assert isinstance(registry, dict)
    assert set(registry) == {"classify_severity", "lookup_incidents", "recommend_action"}
    for name, tool in registry.items():
        assert tool.name == name
        assert callable(tool.run)


def test_describe_tools_renders_every_tool_into_the_prompt(registry: dict[str, Tool]) -> None:
    rendered = describe_tools(registry)
    for name in registry:
        assert name in rendered
    assert "Final Answer" in rendered


def test_build_registry_is_fresh_each_call(incidents_file: Path, fake_llm: object) -> None:
    first = build_registry(
        llm=fake_llm,
        incidents_path=incidents_file,
        lookback_days=90,
        today=TODAY,  # type: ignore[arg-type]
    )
    second = build_registry(
        llm=fake_llm,
        incidents_path=incidents_file,
        lookback_days=90,
        today=TODAY,  # type: ignore[arg-type]
    )
    assert first is not second
    assert first["lookup_incidents"] is not second["lookup_incidents"]
