"""Cache tests (``task02.md`` §8). The cache is shared by both parts, so it is tested once."""

from __future__ import annotations

import contextlib

from triage_core.infra.cache import ResultCache
from triage_core.tools.base import Tool


def test_second_identical_call_is_a_hit_and_runs_the_body_once() -> None:
    cache = ResultCache(enabled=True)
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        return "value"

    first = cache.get_or_compute("lookup_incidents", {"signature": "a"}, compute)
    second = cache.get_or_compute("lookup_incidents", {"signature": "a"}, compute)

    assert not first.hit
    assert second.hit
    assert second.value == "value"
    assert calls == 1
    assert (cache.stats.hits, cache.stats.misses) == (1, 1)


def test_argument_order_does_not_change_the_key() -> None:
    cache = ResultCache(enabled=True)
    cache.get_or_compute("t", {"a": 1, "b": 2}, lambda: "x")
    outcome = cache.get_or_compute("t", {"b": 2, "a": 1}, lambda: "y")
    assert outcome.hit
    assert outcome.value == "x"


def test_different_arguments_miss() -> None:
    cache = ResultCache(enabled=True)
    cache.get_or_compute("t", {"a": 1}, lambda: "x")
    outcome = cache.get_or_compute("t", {"a": 2}, lambda: "y")
    assert not outcome.hit
    assert cache.stats.misses == 2


def test_different_tools_do_not_collide() -> None:
    cache = ResultCache(enabled=True)
    cache.get_or_compute("tool_a", {"a": 1}, lambda: "x")
    outcome = cache.get_or_compute("tool_b", {"a": 1}, lambda: "y")
    assert not outcome.hit
    assert outcome.value == "y"


def test_disabled_cache_counts_misses_and_reruns_the_body() -> None:
    """``--no-cache`` must produce a visibly different summary, not silence."""
    cache = ResultCache(enabled=False)
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        return "value"

    first = cache.get_or_compute("t", {"a": 1}, compute)
    second = cache.get_or_compute("t", {"a": 1}, compute)

    assert not first.hit
    assert not second.hit
    assert calls == 2
    assert (cache.stats.hits, cache.stats.misses) == (0, 2)


def test_a_failing_compute_stores_nothing() -> None:
    cache = ResultCache(enabled=True)
    calls = 0

    def boom() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("nope")

    for _ in range(2):
        with contextlib.suppress(RuntimeError):
            cache.get_or_compute("t", {"a": 1}, boom)

    assert calls == 2


def test_a_fresh_instance_starts_empty() -> None:
    """Injected, not global — every test gets its own (``task02.md`` §4)."""
    first = ResultCache(enabled=True)
    first.get_or_compute("t", {"a": 1}, lambda: "x")
    second = ResultCache(enabled=True)
    outcome = second.get_or_compute("t", {"a": 1}, lambda: "y")
    assert not outcome.hit
    assert second.stats.hits == 0


def test_outcome_reports_timing_and_verdict() -> None:
    """``--demo-cache`` in Part 2 prints both."""
    cache = ResultCache(enabled=True)
    outcome = cache.get_or_compute("t", {"a": 1}, lambda: "x")
    assert outcome.verdict == "MISS"
    assert outcome.elapsed_ms >= 0.0
    assert cache.get_or_compute("t", {"a": 1}, lambda: "x").verdict == "HIT"


def test_enum_and_date_arguments_are_hashable_into_a_key(registry: dict[str, Tool]) -> None:
    """The real call passes a ``Severity``; ``default=str`` must absorb it."""
    from triage_core.domain.models import Severity

    key = ResultCache.key("lookup_incidents", {"severity": Severity.CRITICAL, "signature": "s"})
    assert key.startswith("lookup_incidents:")


def test_the_agent_loop_reuses_a_cached_lookup(run_part1: object) -> None:
    """End to end: the same lookup twice in one process is served from cache."""
    from tests.conftest import POOL_LINE

    cache = ResultCache(enabled=True)
    script = [
        "Thought: classify\nAction: classify_severity\nAction Input: "
        f'{{"log_line": "{POOL_LINE}"}}',
        "Thought: look it up\nAction: lookup_incidents\nAction Input: "
        '{"signature": "db.pool.exhausted:orders-primary", "severity": "critical"}',
        "Thought: again, deliberately\nAction: lookup_incidents\nAction Input: "
        '{"signature": "db.pool.exhausted:orders-primary", "severity": "critical"}',
        'Thought: done\nFinal Answer: {"action": "file_ticket", "severity": "critical",'
        ' "confidence": 0.9, "justification": "j", "matched_incidents": []}',
    ]
    run = run_part1(POOL_LINE, responses=script, cache=cache)  # type: ignore[operator]

    assert run.cache_misses == 1
    assert run.cache_hits == 1
    verdicts = [step.cache for step in run.trace if step.cache is not None]
    assert verdicts == ["miss", "hit"]
