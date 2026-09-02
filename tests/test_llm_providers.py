"""Tests for the optional non-Anthropic provider clients.

All offline (Rule R9): the lazily-imported SDK entry point is monkeypatched with a fake whose
call returns a canned response. The module is skipped entirely when the ``providers`` extra is
not installed, so ``pytest`` stays green on the lean install.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openai", reason="needs the optional 'providers' extra")
pytest.importorskip("groq", reason="needs the optional 'providers' extra")
pytest.importorskip("google.genai", reason="needs the optional 'providers' extra")

from triage_core.llm.client import GeminiLLM, GroqLLM, LLMError, OpenAILLM

SYSTEM = "you are a triage agent"
USER = "Log line: ERROR payments-api pool exhausted"


# ---------------------------------------------------------------------------
# fakes for the OpenAI-shaped SDKs (OpenAI + Groq)
# ---------------------------------------------------------------------------


class _FakeChatClient:
    """Stands in for ``openai.OpenAI(...)`` / ``groq.Groq(...)``."""

    def __init__(self, *, content: str | None = "ok", error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeGeminiClient:
    """Stands in for ``google.genai.Client(...)``."""

    def __init__(self, *, text: str | None = "ok", error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


@pytest.fixture
def patched_openai(monkeypatch: pytest.MonkeyPatch) -> list[_FakeChatClient]:
    """Replace ``openai.OpenAI`` with a factory recording every constructed fake."""
    import openai

    made: list[_FakeChatClient] = []

    def factory(**_: Any) -> _FakeChatClient:
        client = _FakeChatClient(content=factory.content, error=factory.error)  # type: ignore[attr-defined]
        made.append(client)
        return client

    factory.content = "ok"  # type: ignore[attr-defined]
    factory.error = None  # type: ignore[attr-defined]
    monkeypatch.setattr(openai, "OpenAI", factory)
    factory.made = made  # type: ignore[attr-defined]
    return factory  # type: ignore[return-value]


@pytest.fixture
def patched_groq(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace ``groq.Groq`` with a factory recording every constructed fake."""
    import groq

    def factory(**_: Any) -> _FakeChatClient:
        client = _FakeChatClient(content=factory.content, error=factory.error)  # type: ignore[attr-defined]
        factory.made.append(client)  # type: ignore[attr-defined]
        return client

    factory.content = "ok"  # type: ignore[attr-defined]
    factory.error = None  # type: ignore[attr-defined]
    factory.made = []  # type: ignore[attr-defined]
    monkeypatch.setattr(groq, "Groq", factory)
    return factory


@pytest.fixture
def patched_gemini(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace ``google.genai.Client`` with a factory recording every constructed fake."""
    from google import genai

    def factory(**_: Any) -> _FakeGeminiClient:
        client = _FakeGeminiClient(text=factory.text, error=factory.error)  # type: ignore[attr-defined]
        factory.made.append(client)  # type: ignore[attr-defined]
        return client

    factory.text = "ok"  # type: ignore[attr-defined]
    factory.error = None  # type: ignore[attr-defined]
    factory.made = []  # type: ignore[attr-defined]
    monkeypatch.setattr(genai, "Client", factory)
    return factory


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_returns_the_message_content(patched_openai: Any) -> None:
    patched_openai.content = "Thought: t\nAction: classify_severity\nAction Input: {}"
    llm = OpenAILLM("sk-not-real", model="gpt-4o-mini")
    assert llm.complete(SYSTEM, USER).startswith("Thought:")

    sent = patched_openai.made[0].calls[0]
    assert sent["model"] == "gpt-4o-mini"
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM}
    assert sent["messages"][1] == {"role": "user", "content": USER}
    assert sent["temperature"] == 0.0


def test_openai_omits_temperature_when_none(patched_openai: Any) -> None:
    llm = OpenAILLM("sk-not-real", model="o3-mini", temperature=None)
    llm.complete(SYSTEM, USER)
    assert "temperature" not in patched_openai.made[0].calls[0]


def test_openai_wraps_transport_errors(patched_openai: Any) -> None:
    patched_openai.error = RuntimeError("connection reset")
    llm = OpenAILLM("sk-not-real", model="gpt-4o-mini")
    with pytest.raises(LLMError, match="OpenAI request failed"):
        llm.complete(SYSTEM, USER)


def test_openai_raises_on_empty_content(patched_openai: Any) -> None:
    patched_openai.content = "   "
    llm = OpenAILLM("sk-not-real", model="gpt-4o-mini")
    with pytest.raises(LLMError, match="OpenAI returned no text"):
        llm.complete(SYSTEM, USER)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


def test_groq_returns_the_message_content(patched_groq: Any) -> None:
    patched_groq.content = "Final Answer: {}"
    llm = GroqLLM("gsk-not-real", model="llama-3.3-70b-versatile")
    assert llm.complete(SYSTEM, USER) == "Final Answer: {}"
    assert patched_groq.made[0].calls[0]["model"] == "llama-3.3-70b-versatile"


def test_groq_wraps_errors_and_empty(patched_groq: Any) -> None:
    patched_groq.error = RuntimeError("429 rate limited")
    llm = GroqLLM("gsk-not-real", model="llama-3.3-70b-versatile")
    with pytest.raises(LLMError, match="Groq request failed"):
        llm.complete(SYSTEM, USER)

    patched_groq.error = None
    patched_groq.content = None
    with pytest.raises(LLMError, match="Groq returned no text"):
        GroqLLM("gsk-not-real", model="llama-3.3-70b-versatile").complete(SYSTEM, USER)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_returns_the_response_text(patched_gemini: Any) -> None:
    patched_gemini.text = "Thought: classify first\nAction: classify_severity\nAction Input: {}"
    llm = GeminiLLM("g-not-real", model="gemini-2.0-flash")
    assert llm.complete(SYSTEM, USER).startswith("Thought:")

    sent = patched_gemini.made[0].calls[0]
    assert sent["model"] == "gemini-2.0-flash"
    assert sent["contents"] == USER
    assert sent["config"].system_instruction == SYSTEM
    assert sent["config"].temperature == 0.0
    assert sent["config"].max_output_tokens == 16000


def test_gemini_omits_temperature_when_none(patched_gemini: Any) -> None:
    llm = GeminiLLM("g-not-real", model="gemini-2.0-flash", temperature=None)
    llm.complete(SYSTEM, USER)
    assert patched_gemini.made[0].calls[0]["config"].temperature is None


def test_gemini_wraps_errors_and_empty(patched_gemini: Any) -> None:
    patched_gemini.error = RuntimeError("permission denied")
    with pytest.raises(LLMError, match="Gemini request failed"):
        GeminiLLM("g-not-real", model="gemini-2.0-flash").complete(SYSTEM, USER)

    patched_gemini.error = None
    patched_gemini.text = None
    with pytest.raises(LLMError, match="Gemini returned no text"):
        GeminiLLM("g-not-real", model="gemini-2.0-flash").complete(SYSTEM, USER)


# ---------------------------------------------------------------------------
# all three satisfy the shared contract
# ---------------------------------------------------------------------------


def test_every_provider_client_is_a_text_completer(
    patched_openai: Any, patched_groq: Any, patched_gemini: Any
) -> None:
    from triage_core.domain.ports import TextCompleter

    for llm in (
        OpenAILLM("x", model="gpt-4o-mini"),
        GroqLLM("x", model="llama-3.3-70b-versatile"),
        GeminiLLM("x", model="gemini-2.0-flash"),
    ):
        assert isinstance(llm, TextCompleter)
