"""CLI and configuration tests.

Covers the offline fallback (Rule R7), the trace being printed at all (Rule R8), and the exit
codes the run modes promise.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from tests.conftest import REAL_INCIDENTS, SAMPLES

from react_from_scratch.cli import EXIT_OK, EXIT_PARTIAL, EXIT_USAGE, main
from triage_core.config import DEFAULT_TODAY, PROVIDER_KEYS, Settings, build_llm, load_settings
from triage_core.llm.client import AnthropicLLM, FakeLLM

pytestmark = pytest.mark.usefixtures("_no_real_key")


@pytest.fixture
def _no_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real key or provider choice leak into the suite (Rule R9)."""
    for env_key in PROVIDER_KEYS.values():
        monkeypatch.delenv(env_key, raising=False)
    for name in ("TRIAGE_PROVIDER", "TRIAGE_MODEL", "TRIAGE_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_build_llm_falls_back_to_the_fake_with_one_warning() -> None:
    stream = io.StringIO()
    llm = build_llm(Settings(anthropic_api_key=None), stream=stream)

    assert isinstance(llm, FakeLLM)
    warning = stream.getvalue()
    assert warning.count("\n") == 1
    assert "no LLM API key found" in warning
    assert "FakeLLM" in warning


def test_build_llm_honours_force_fake_even_with_a_key() -> None:
    stream = io.StringIO()
    llm = build_llm(Settings(anthropic_api_key="sk-ant-not-real"), force_fake=True, stream=stream)
    assert isinstance(llm, FakeLLM)
    assert "--fake" in stream.getvalue()


def test_build_llm_selects_the_real_client_when_a_key_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No network: the SDK client is constructed but never called."""
    llm = build_llm(Settings(anthropic_api_key="sk-ant-not-real"), stream=io.StringIO())
    assert isinstance(llm, AnthropicLLM)


def test_load_settings_reads_overrides_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_MAX_ITERATIONS", "9")
    monkeypatch.setenv("TRIAGE_LOOKBACK_DAYS", "30")
    settings = load_settings()
    assert settings.max_iterations == 9
    assert settings.lookback_days == 30

    overridden = load_settings(max_iterations=2)
    assert overridden.max_iterations == 2


def test_load_settings_ignores_unparseable_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRIAGE_MAX_ITERATIONS", "not-a-number")
    monkeypatch.setenv("TRIAGE_TODAY", "the day before yesterday")
    settings = load_settings()
    assert settings.max_iterations == 6
    assert settings.today == DEFAULT_TODAY


def test_settings_has_api_key_treats_blank_as_absent() -> None:
    assert not Settings(anthropic_api_key="   ").has_api_key
    assert not Settings(anthropic_api_key=None).has_api_key
    assert Settings(anthropic_api_key="sk-ant-x").has_api_key


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_a_full_trace_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--file", str(SAMPLES / "pool_exhausted.log"), "--fake"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "ReAct (raw Python)" in out
    assert out.count("Thought") >= 4
    assert "Action" in out
    assert "Observ." in out
    assert "── Result" in out
    assert "Iterations    : 4" in out
    assert "Cache:" in out


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
    """The shipped data and the shipped samples agree with the documented policy table."""
    code = main(["--file", str(SAMPLES / sample), "--fake", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["decision"]["action"] == action
    assert payload["decision"]["complete"] is True


def test_cli_accepts_a_positional_log_line(capsys: pytest.CaptureFixture[str]) -> None:
    line = "2026-08-30T00:00:01Z INFO catalog-api  Warming cache after deploy (entries=1)"
    code = main([line, "--fake", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["decision"]["action"] == "ignore"


def test_cli_reports_a_partial_result_with_exit_code_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--file", str(SAMPLES / "pool_exhausted.log"), "--fake", "--max-iters", "2"])
    out = capsys.readouterr().out

    assert code == EXIT_PARTIAL
    assert "Incomplete" in out
    assert "iteration_limit" in out


def test_cli_no_cache_flag_produces_no_hits(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--file", str(SAMPLES / "pool_exhausted.log"), "--fake", "--no-cache", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_hits"] == 0
    assert payload["cache_misses"] == 1


def test_cli_rejects_a_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--file", "does-not-exist.log", "--fake"])
    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_an_empty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.log"
    empty.write_text("\n  \n", encoding="utf-8")
    code = main(["--file", str(empty), "--fake"])
    assert code == EXIT_USAGE
    assert "no non-blank line" in capsys.readouterr().err


def test_cli_reports_a_bad_incident_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "incidents.json"
    broken.write_text("not json", encoding="utf-8")
    code = main(
        ["--file", str(SAMPLES / "pool_exhausted.log"), "--fake", "--incidents", str(broken)]
    )
    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_cli_reports_a_provider_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad key / unknown model / network drop is a clean message, not a traceback."""
    from triage_core.llm.client import LLMError

    class _Broken:
        def complete(self, system: str, user: str) -> str:
            raise LLMError("401 - invalid api key")

    monkeypatch.setattr("react_from_scratch.cli.build_llm", lambda *a, **k: _Broken())
    code = main(["--file", str(SAMPLES / "pool_exhausted.log")])

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "LLM request failed" in err
    assert "invalid api key" in err


def test_shipped_incident_data_loads() -> None:
    """The demo data must stay valid — the CLI depends on it by default."""
    from triage_core.tools.incident_lookup import load_incidents

    incidents = load_incidents(REAL_INCIDENTS)
    assert len(incidents) >= 12
    assert len({i.signature for i in incidents}) >= 5
