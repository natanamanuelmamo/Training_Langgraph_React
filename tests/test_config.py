"""Provider auto-detection and ``build_llm`` dispatch (Rule R14).

Offline (Rule R9): the ``build_llm`` cases construct the client but never call ``.complete()``,
so no SDK entry point is exercised and no network is touched.
"""

from __future__ import annotations

import io

import pytest

from triage_core.config import (
    PROVIDER_KEYS,
    Settings,
    _detect_provider,
    build_llm,
    load_settings,
)
from triage_core.llm.client import AnthropicLLM, FakeLLM

_TRIAGE_VARS = ("TRIAGE_PROVIDER", "TRIAGE_MODEL", "TRIAGE_TEMPERATURE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no provider key and no provider override set."""
    for env_key in PROVIDER_KEYS.values():
        monkeypatch.delenv(env_key, raising=False)
    for name in _TRIAGE_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# _detect_provider
# ---------------------------------------------------------------------------


def test_defaults_to_anthropic_when_nothing_is_set() -> None:
    assert _detect_provider() == "anthropic"


@pytest.mark.parametrize(
    ("env_key", "provider"),
    [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("GEMINI_API_KEY", "gemini"),
        ("GROQ_API_KEY", "groq"),
    ],
)
def test_detects_the_provider_from_its_key(
    monkeypatch: pytest.MonkeyPatch, env_key: str, provider: str
) -> None:
    monkeypatch.setenv(env_key, "not-a-real-key")
    assert _detect_provider() == provider


def test_priority_order_when_several_keys_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert _detect_provider() == "openai"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert _detect_provider() == "anthropic"


def test_triage_provider_overrides_key_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("TRIAGE_PROVIDER", "groq")
    assert _detect_provider() == "groq"


def test_unknown_triage_provider_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("TRIAGE_PROVIDER", "cohere")
    assert _detect_provider() == "openai"


def test_blank_key_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert _detect_provider() == "anthropic"


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------


def test_load_settings_picks_the_provider_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-x")
    settings = load_settings()
    assert settings.provider == "groq"
    assert settings.model == "llama-3.3-70b-versatile"
    assert settings.llm_api_key == "gsk-x"
    assert settings.anthropic_api_key is None


def test_explicit_triage_model_wins_over_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("TRIAGE_MODEL", "gpt-4o")
    assert load_settings().model == "gpt-4o"


def test_load_settings_reads_anthropic_key_into_both_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    settings = load_settings()
    assert settings.provider == "anthropic"
    assert settings.anthropic_api_key == "sk-ant-x"
    assert settings.resolved_key == "sk-ant-x"
    assert settings.model == "claude-opus-5"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0.0), ("0.7", 0.7), ("nonsense", 0.0)],
)
def test_temperature_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("TRIAGE_TEMPERATURE", raw)
    assert load_settings().temperature == expected


def test_empty_temperature_means_omit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("TRIAGE_TEMPERATURE", "")
    assert load_settings().temperature is None


def test_unset_temperature_keeps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert load_settings().temperature == 0.0


# ---------------------------------------------------------------------------
# build_llm dispatch
# ---------------------------------------------------------------------------


def test_build_llm_returns_the_fake_when_the_selected_provider_has_no_key() -> None:
    stream = io.StringIO()
    llm = build_llm(Settings(provider="openai", llm_api_key=None), stream=stream)
    assert isinstance(llm, FakeLLM)
    assert "no LLM API key found" in stream.getvalue()


def test_build_llm_dispatches_to_anthropic() -> None:
    llm = build_llm(
        Settings(provider="anthropic", anthropic_api_key="sk-ant-x"), stream=io.StringIO()
    )
    assert isinstance(llm, AnthropicLLM)


@pytest.mark.parametrize(
    ("provider", "model", "class_name"),
    [
        ("openai", "gpt-4o-mini", "OpenAILLM"),
        ("groq", "llama-3.3-70b-versatile", "GroqLLM"),
        ("gemini", "gemini-2.0-flash", "GeminiLLM"),
    ],
)
def test_build_llm_dispatches_to_each_provider(provider: str, model: str, class_name: str) -> None:
    pytest.importorskip(
        {"openai": "openai", "groq": "groq", "gemini": "google.genai"}[provider],
        reason="needs the optional 'providers' extra",
    )
    llm = build_llm(
        Settings(provider=provider, llm_api_key="key-not-real", model=model),
        stream=io.StringIO(),
    )
    assert type(llm).__name__ == class_name


def test_has_api_key_is_provider_aware() -> None:
    assert Settings(provider="anthropic", anthropic_api_key="x").has_api_key
    assert not Settings(provider="anthropic", llm_api_key="x").has_api_key
    assert Settings(provider="groq", llm_api_key="x").has_api_key
    assert not Settings(provider="groq", anthropic_api_key="x").has_api_key
    assert not Settings(provider="openai", llm_api_key="   ").has_api_key
