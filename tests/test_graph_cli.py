"""Part 2 CLI tests (``task02.md`` §6).

Covers every run mode a reviewer is told to try, and the exit codes they promise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="Part 2 requires the optional 'graph' extra")

from tests.conftest import SAMPLES

from react_langgraph.cli import (
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_USAGE,
    main,
)
from triage_core.config import PROVIDER_KEYS

pytestmark = pytest.mark.graph


@pytest.fixture(autouse=True)
def _no_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real key or provider choice leak into the suite (Rule R9)."""
    for env_key in PROVIDER_KEYS.values():
        monkeypatch.delenv(env_key, raising=False)
    for name in ("TRIAGE_PROVIDER", "TRIAGE_MODEL", "TRIAGE_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)


def _args(sample: str, *extra: str) -> list[str]:
    return ["--file", str(SAMPLES / sample), "--fake", *extra]


def test_cli_prints_node_transitions_and_the_loop_backs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(_args("pool_exhausted.log", "--thread-id", "t1"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "LangGraph ReAct" in out
    assert "thread_id: t1" in out
    assert out.count("▸ reason") == 4
    assert out.count("▸ act") == 3
    assert "▸ decide" in out
    assert "CACHE MISS" in out
    assert "── Result" in out


def test_cli_auto_approve_runs_the_sensitive_node(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(_args("cert_expired.log", "--auto-approve", "--thread-id", "t2"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "PAUSED before node 'page_on_call'" in out
    assert "▸ page_on_call" in out
    assert "paged: sre-primary" in out
    assert "Action        : page_on_call" in out


def test_cli_auto_reject_changes_the_outcome(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        _args(
            "cert_expired.log",
            "--auto-reject",
            "--note",
            "renewal in flight",
            "--thread-id",
            "t3",
        )
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "PAUSED before node 'page_on_call'" in out
    assert "▸ page_on_call" not in out
    assert "page rejected by human" in out
    assert "Action        : file_ticket" in out
    assert "renewal in flight" in out


@pytest.mark.parametrize(
    ("sample", "action"),
    [
        ("pool_exhausted.log", "file_ticket"),
        ("oom_recurring.log", "page_on_call"),
        ("cert_expired.log", "page_on_call"),
        ("disk_full_unresolved.log", "page_on_call"),
        ("slow_query.log", "file_ticket"),
        ("cache_warm.log", "ignore"),
    ],
)
def test_cli_decides_correctly_for_every_shipped_sample(
    capsys: pytest.CaptureFixture[str], sample: str, action: str
) -> None:
    """Part 2 reaches the same verdicts Part 1 does on the shipped data."""
    code = main(_args(sample, "--auto-approve", "--json", "--thread-id", f"s-{sample}"))
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["decision"]["action"] == action


def test_cli_non_sensitive_paths_never_pause(capsys: pytest.CaptureFixture[str]) -> None:
    for sample in ("slow_query.log", "cache_warm.log"):
        main(_args(sample, "--thread-id", f"quiet-{sample}"))
        out = capsys.readouterr().out
        assert "PAUSED" not in out
        assert "▸ notify" in out


def test_cli_demo_cache_shows_a_miss_then_a_hit(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_args("pool_exhausted.log", "--demo-cache"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "run 1  lookup_incidents  MISS" in out
    assert "run 2  lookup_incidents  HIT" in out
    assert "cache: 1 hit / 1 miss" in out


def test_cli_demo_cache_with_no_cache_shows_two_misses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(_args("pool_exhausted.log", "--demo-cache", "--no-cache"))
    out = capsys.readouterr().out

    assert "run 2  lookup_incidents  MISS" in out
    assert "cache: 0 hit / 2 miss" in out


def test_cli_print_graph_shows_the_loop_back_and_interrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--print-graph", "--fake"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "```mermaid" in out
    assert "act --> reason" in out
    assert "page_on_call" in out
    assert "interrupt" in out


def test_cli_resume_shows_the_pause_then_settles_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(_args("cert_expired.log", "--resume", "inc-resume", "--auto-approve"))
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "[phase 1]" in out
    assert "next = ('page_on_call',)" in out
    assert "[phase 2]" in out
    assert "next = ()" in out


def test_cli_partial_result_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_args("pool_exhausted.log", "--max-iters", "2", "--thread-id", "short"))
    out = capsys.readouterr().out

    assert code == EXIT_PARTIAL
    assert "▸ halt" in out
    assert "Incomplete" in out


def test_cli_rejects_a_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--file", "nope.log", "--fake"])
    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_cli_reports_a_provider_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A provider misconfiguration surfaces as a clean message, not a traceback."""
    from triage_core.llm.client import LLMError

    class _Broken:
        def complete(self, system: str, user: str) -> str:
            raise LLMError("model 'claude-opus-5' not found on this provider")

    monkeypatch.setattr("react_langgraph.cli.build_llm", lambda *a, **k: _Broken())
    code = main(["--file", str(SAMPLES / "pool_exhausted.log"), "--thread-id", "boom"])

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "LLM request failed" in err
    assert "not found on this provider" in err


def test_cli_rejects_an_empty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.log"
    empty.write_text("\n \n", encoding="utf-8")
    code = main(["--file", str(empty), "--fake"])
    assert code == EXIT_USAGE
    assert "no non-blank line" in capsys.readouterr().err


def test_cli_accepts_a_positional_log_line(capsys: pytest.CaptureFixture[str]) -> None:
    line = "2026-08-30T06:00:02Z INFO catalog-api  Warming cache after deploy (entries=1)"
    code = main([line, "--fake", "--json", "--thread-id", "pos"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["decision"]["action"] == "ignore"


def test_cli_falls_back_to_the_fake_llm_without_a_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rule R7: Part 2 is demonstrable offline too."""
    main(["--file", str(SAMPLES / "cache_warm.log"), "--thread-id", "nokey"])
    err = capsys.readouterr().err
    assert "no LLM API key found" in err
