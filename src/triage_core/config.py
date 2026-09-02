"""Settings loading and LLM selection.

Serves both parts. Configuration comes from environment variables only — no secret is ever
committed (Rule R7). The provider is auto-detected from whichever API key is present
(``TRIAGE_PROVIDER`` forces a choice); when no key is found the deterministic
:class:`~triage_core.llm.client.FakeLLM` is selected and a single warning line is printed, so
the whole project stays demonstrable and testable offline.

Adding a provider is one branch in :func:`build_llm` plus one row in :data:`PROVIDER_KEYS` and
:data:`DEFAULT_MODELS` — nothing else changes (Rule R14).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from triage_core.domain.ports import TextCompleter
from triage_core.llm.client import AnthropicLLM, FakeLLM, GeminiLLM, GroqLLM, OpenAILLM

#: The incident history is a fixed historical snapshot dated around August 2026. Anchoring
#: "today" to a constant keeps the lookback window meaningful however long from now the
#: project is graded, and keeps every demo run deterministic (Rule R9). Override with
#: ``TRIAGE_TODAY`` to triage against the real current date.
DEFAULT_TODAY = date(2026, 8, 30)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Provider name -> the environment variable holding its API key. Iteration order is the
#: auto-detection priority when several keys are set.
PROVIDER_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

#: Best-effort default model per provider, used when ``TRIAGE_MODEL`` is unset. These drift;
#: set ``TRIAGE_MODEL`` explicitly for anything other than a quick check.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything both parts need to run, resolved from the environment."""

    provider: str = "anthropic"
    anthropic_api_key: str | None = None
    llm_api_key: str | None = None
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    effort: str = "low"
    temperature: float | None = 0.0
    max_iterations: int = 6
    lookback_days: int = 90
    today: date = DEFAULT_TODAY
    incidents_path: Path = _REPO_ROOT / "data" / "incidents.json"
    cache_enabled: bool = True

    @property
    def resolved_key(self) -> str | None:
        """The API key for the selected provider (``anthropic_api_key`` when on Anthropic)."""
        if self.provider == "anthropic":
            return self.anthropic_api_key
        return self.llm_api_key

    @property
    def has_api_key(self) -> bool:
        """Whether a usable key was found for the selected provider."""
        key = self.resolved_key
        return bool(key and key.strip())


def _env_str(name: str) -> str | None:
    """Read a string environment variable, treating blank as absent."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _env_int(name: str, fallback: int) -> int:
    """Read an integer environment variable, falling back on absence or garbage."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _env_date(name: str, fallback: date) -> date:
    """Read an ISO date environment variable, falling back on absence or garbage."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return fallback


def _env_temperature(name: str, fallback: float | None) -> float | None:
    """Read a temperature.

    An explicitly empty value (``TRIAGE_TEMPERATURE=``) means "omit the parameter", which some
    reasoning models require; an unset variable keeps the fallback; garbage keeps the fallback.
    """
    if name not in os.environ:
        return fallback
    raw = os.environ[name].strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return fallback


def _detect_provider() -> str:
    """Work out which provider to use.

    ``TRIAGE_PROVIDER`` wins if it names a known provider. Otherwise the first provider in
    :data:`PROVIDER_KEYS` whose key is set is chosen. Falls back to ``"anthropic"`` (which then
    lands on :class:`FakeLLM` if that key is also absent).
    """
    explicit = _env_str("TRIAGE_PROVIDER")
    if explicit and explicit.lower() in PROVIDER_KEYS:
        return explicit.lower()
    for provider, env_key in PROVIDER_KEYS.items():
        if _env_str(env_key):
            return provider
    return "anthropic"


def load_settings(**overrides: Any) -> Settings:
    """Build a :class:`Settings` from ``.env``, the environment, then explicit overrides.

    Args:
        **overrides: Field values that win over the environment — how CLI flags are applied.

    Returns:
        The resolved settings.
    """
    load_dotenv()
    provider = _detect_provider()
    anthropic_key = _env_str("ANTHROPIC_API_KEY")
    provider_key = _env_str(PROVIDER_KEYS[provider])

    settings = Settings(
        provider=provider,
        anthropic_api_key=anthropic_key,
        llm_api_key=provider_key,
        model=_env_str("TRIAGE_MODEL") or DEFAULT_MODELS[provider],
        max_tokens=_env_int("TRIAGE_MAX_TOKENS", 16000),
        effort=_env_str("TRIAGE_EFFORT") or "low",
        temperature=_env_temperature("TRIAGE_TEMPERATURE", 0.0),
        max_iterations=_env_int("TRIAGE_MAX_ITERATIONS", 6),
        lookback_days=_env_int("TRIAGE_LOOKBACK_DAYS", 90),
        today=_env_date("TRIAGE_TODAY", DEFAULT_TODAY),
    )
    return replace(settings, **overrides) if overrides else settings


def build_llm(
    settings: Settings,
    *,
    force_fake: bool = False,
    stream: Any = None,
) -> TextCompleter:
    """Select the reasoning back end for this run.

    Args:
        settings: Resolved settings.
        force_fake: Use the offline fake even when a key is present (the ``--fake`` flag).
        stream: Where the warning goes. Defaults to stderr.

    Returns:
        A completer: the real client for the selected provider when its key is available, the
        deterministic fake otherwise.
    """
    out = stream if stream is not None else sys.stderr

    if force_fake:
        print("[info] --fake given — using the deterministic FakeLLM.", file=out)
        return FakeLLM()

    if not settings.has_api_key:
        print(
            "[warn] no LLM API key found "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY) — "
            "using the deterministic FakeLLM. Traces are illustrative, not live.",
            file=out,
        )
        return FakeLLM()

    key = settings.resolved_key
    assert key is not None  # narrowed by has_api_key

    if settings.provider == "anthropic":
        return AnthropicLLM(
            key,
            model=settings.model,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
        )
    if settings.provider == "openai":
        return OpenAILLM(
            key,
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    if settings.provider == "groq":
        return GroqLLM(
            key,
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    if settings.provider == "gemini":
        return GeminiLLM(
            key,
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )

    # Unreachable: _detect_provider only returns keys of PROVIDER_KEYS.
    raise ValueError(f"unknown provider {settings.provider!r}")
