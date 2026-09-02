"""Shared fixtures. Every test is deterministic and offline (Rule R9)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest

from react_from_scratch.agent import run_react_agent
from triage_core.domain.models import TriageRun
from triage_core.infra.cache import ResultCache
from triage_core.infra.tracing import Tracer
from triage_core.llm.client import FakeLLM, ScriptedLLM
from triage_core.tools.base import Tool
from triage_core.tools.registry import build_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_INCIDENTS = REPO_ROOT / "data" / "incidents.json"
SAMPLES = REPO_ROOT / "data" / "samples"

#: Matches ``TRIAGE_TODAY`` / ``config.DEFAULT_TODAY`` so the seeded lookback stays meaningful.
TODAY = date(2026, 8, 30)

POOL_LINE = (
    "2026-08-29T03:14:07Z ERROR payments-api  Connection pool exhausted after 30s "
    "(db=orders-primary, active=100/100, waiting=482)"
)
OOM_LINE = (
    "2026-08-29T04:41:55Z ERROR recommendation-worker  Container killed: out of memory "
    "(limit=4Gi, rss=4.1Gi)"
)
CERT_LINE = (
    "2026-08-30T09:02:11Z ERROR checkout-gateway  TLS handshake failed: server certificate "
    "has expired (host=checkout.internal)"
)
WARN_LINE = (
    "2026-08-29T18:07:33Z WARN search-api  Upstream request timed out after 2000ms "
    "(endpoint=/suggest)"
)
INFO_LINE = "2026-08-30T06:00:02Z INFO catalog-api  Warming cache after deploy (entries=18422)"


@pytest.fixture
def incidents_file(tmp_path: Path) -> Path:
    """A small incident history covering every policy branch.

    Deliberately separate from ``data/incidents.json`` so policy tests do not break when the
    demo data is edited. ``INC-0009`` sits outside the 90-day lookback to prove the filter runs.
    """
    records = [
        # critical + resolved priors, 2 occurrences -> critical_known_fix
        {
            "incident_id": "INC-0001",
            "signature": "db.pool.exhausted:orders-primary",
            "severity": "critical",
            "opened_at": "2026-07-14",
            "resolved": True,
            "resolution": "raised pool to 200",
            "occurrences": 1,
        },
        {
            "incident_id": "INC-0002",
            "signature": "db.pool.exhausted:orders-primary",
            "severity": "critical",
            "opened_at": "2026-08-02",
            "resolved": True,
            "resolution": "killed a report query",
            "occurrences": 1,
        },
        # critical + 4 occurrences -> critical_regression
        {
            "incident_id": "INC-0003",
            "signature": "oom.killed:recommendation-worker",
            "severity": "critical",
            "opened_at": "2026-07-19",
            "resolved": True,
            "resolution": "raised memory limit",
            "occurrences": 2,
        },
        {
            "incident_id": "INC-0004",
            "signature": "oom.killed:recommendation-worker",
            "severity": "critical",
            "opened_at": "2026-08-21",
            "resolved": False,
            "resolution": None,
            "occurrences": 2,
        },
        # critical + unresolved prior -> critical_unresolved
        {
            "incident_id": "INC-0005",
            "signature": "disk.full:log-shipper",
            "severity": "critical",
            "opened_at": "2026-08-11",
            "resolved": False,
            "resolution": None,
            "occurrences": 1,
        },
        # warning path
        {
            "incident_id": "INC-0006",
            "signature": "net.timeout:search-api",
            "severity": "warning",
            "opened_at": "2026-07-28",
            "resolved": True,
            "resolution": "raised the read timeout",
            "occurrences": 1,
        },
        # info path
        {
            "incident_id": "INC-0007",
            "signature": "cache.warm:catalog-api",
            "severity": "info",
            "opened_at": "2026-08-27",
            "resolved": True,
            "resolution": "expected at deploy time",
            "occurrences": 1,
        },
        # outside the lookback window — must never be returned
        {
            "incident_id": "INC-0009",
            "signature": "db.pool.exhausted:orders-primary",
            "severity": "critical",
            "opened_at": "2026-01-12",
            "resolved": True,
            "resolution": "too old to count",
            "occurrences": 9,
        },
    ]
    path = tmp_path / "incidents.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


@pytest.fixture
def fake_llm() -> FakeLLM:
    """The offline state-machine completer."""
    return FakeLLM()


@pytest.fixture
def registry(incidents_file: Path, fake_llm: FakeLLM) -> dict[str, Tool]:
    """A fresh registry over the test incident history."""
    return build_registry(
        llm=fake_llm, incidents_path=incidents_file, lookback_days=90, today=TODAY
    )


@pytest.fixture
def fresh_cache() -> ResultCache:
    """An enabled, empty cache."""
    return ResultCache(enabled=True)


@pytest.fixture
def scripted() -> Callable[..., ScriptedLLM]:
    """Build a :class:`ScriptedLLM` from positional responses."""

    def _build(*responses: str) -> ScriptedLLM:
        return ScriptedLLM(responses)

    return _build


@pytest.fixture
def run_part1(registry: dict[str, Tool], fresh_cache: ResultCache) -> Callable[..., TriageRun]:
    """Run Part 1's loop against the test registry."""

    def _run(
        log_line: str,
        llm: object | None = None,
        *,
        responses: Sequence[str] | None = None,
        max_iterations: int = 6,
        cache: ResultCache | None = None,
    ) -> TriageRun:
        completer = llm if llm is not None else ScriptedLLM(responses or [])
        return run_react_agent(
            log_line,
            llm=completer,  # type: ignore[arg-type]
            tools=registry,
            cache=cache if cache is not None else fresh_cache,
            tracer=Tracer(),
            max_iterations=max_iterations,
        )

    return _run
