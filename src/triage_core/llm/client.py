"""The reasoning step's back end: real provider clients and two offline stand-ins.

Serves both parts. Every class here satisfies :class:`~triage_core.domain.ports.TextCompleter`
structurally, so nothing that consumes them knows which one it has (Rule R14).

* :class:`AnthropicLLM` — the assignment's primary model, on its native SDK.
* :class:`OpenAILLM`, :class:`GroqLLM`, :class:`GeminiLLM` — added at the user's request so a
  key swap in ``.env`` changes provider (see ``CLAUDE.md`` §7). Each uses that provider's
  official SDK, lazy-imported inside ``__init__`` so the SDK is only needed when the provider
  is actually selected, and each raises :class:`LLMError` on any failure — the same contract
  as ``AnthropicLLM``.
* :class:`ScriptedLLM` — replays an explicit list, for deterministic tests (Rule R9).
* :class:`FakeLLM` — a deterministic state machine over the transcript, so ``--fake`` works on
  any log line and the project is demonstrable offline with no key (Rule R7).

Two fakes rather than one because they do different jobs: a fixed script gives tests exact
control but breaks on unseen input; a state machine handles unseen input but cannot express
"now return malformed output on turn 3".
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Literal, get_args

from triage_core.domain.errors import TriageError

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

#: Effort levels the Messages API accepts inside ``output_config``.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
VALID_EFFORTS: tuple[str, ...] = get_args(Effort)


class LLMError(TriageError):
    """The model could not be reached, or refused the request."""


class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API.

    The only external SDK Part 1 is allowed to touch (Rule R1).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 16000,
        effort: str = "low",
    ) -> None:
        """Build a client.

        Args:
            api_key: Anthropic API key.
            model: Model id. Opus 5 is the current default.
            max_tokens: Output cap. Adaptive thinking tokens count against this, so it is
                deliberately generous — an unused cap costs nothing.
            effort: ``low`` by default. Each reason step is a small "which tool next"
                decision repeated up to ``MAX_ITERATIONS`` times, which is what low effort is
                for. An unrecognised value falls back to ``low`` rather than failing the run
                on a typo in an environment variable.
        """
        import anthropic  # imported lazily so the offline path needs no SDK at import time

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._effort: Effort = effort if effort in VALID_EFFORTS else "low"  # type: ignore[assignment]

    def complete(self, system: str, user: str) -> str:
        """Send one request and return the text of the response.

        Args:
            system: The system prompt.
            user: The rendered transcript.

        Returns:
            The concatenated text blocks of the reply, thinking blocks skipped.

        Raises:
            LLMError: The request failed or the model declined it.
        """
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMError(f"model declined the request (stop_details={details})")

        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ]
        text = "\n".join(parts).strip()
        if not text:
            raise LLMError("model returned no text content")
        return text


def _require(text: str | None, provider: str) -> str:
    """Return non-empty response text, or raise the shared :class:`LLMError`.

    Args:
        text: The text the provider returned, possibly ``None`` or blank.
        provider: Provider name, for the error message.

    Returns:
        The stripped text.

    Raises:
        LLMError: The provider returned nothing usable.
    """
    stripped = (text or "").strip()
    if not stripped:
        raise LLMError(f"{provider} returned no text content")
    return stripped


class OpenAILLM:
    """Wrapper over the OpenAI Chat Completions API.

    Requires the optional ``openai`` dependency (``pip install -e ".[providers]"``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_tokens: int = 16000,
        temperature: float | None = 0.0,
    ) -> None:
        """Build a client.

        Args:
            api_key: OpenAI API key.
            model: Model id, e.g. ``gpt-4o-mini``.
            max_tokens: Output cap.
            temperature: Sampling temperature; ``None`` omits the parameter, which some
                reasoning models require.
        """
        import openai  # lazy: only needed when this provider is selected

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Send one request and return the assistant text.

        Args:
            system: The system prompt.
            user: The rendered transcript.

        Returns:
            The assistant message content.

        Raises:
            LLMError: The request failed or returned no text.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return _require(response.choices[0].message.content, "OpenAI")


class GroqLLM:
    """Wrapper over the Groq Chat Completions API (OpenAI-shaped).

    Requires the optional ``groq`` dependency (``pip install -e ".[providers]"``).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_tokens: int = 16000,
        temperature: float | None = 0.0,
    ) -> None:
        """Build a client.

        Args:
            api_key: Groq API key.
            model: Model id, e.g. ``llama-3.3-70b-versatile``.
            max_tokens: Output cap.
            temperature: Sampling temperature; ``None`` omits the parameter.
        """
        from groq import Groq  # lazy: only needed when this provider is selected

        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Send one request and return the assistant text.

        Args:
            system: The system prompt.
            user: The rendered transcript.

        Returns:
            The assistant message content.

        Raises:
            LLMError: The request failed or returned no text.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        return _require(response.choices[0].message.content, "Groq")


class GeminiLLM:
    """Wrapper over the Google Gemini API via the official ``google-genai`` SDK.

    Requires the optional ``google-genai`` dependency (``pip install -e ".[providers]"``).
    Deliberately not ``langchain-google-genai`` — that imports ``langchain_core``, which
    Rule R1 bans.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_tokens: int = 16000,
        temperature: float | None = 0.0,
    ) -> None:
        """Build a client.

        Args:
            api_key: Gemini API key.
            model: Model id, e.g. ``gemini-2.0-flash``.
            max_tokens: Output cap (mapped to ``max_output_tokens``).
            temperature: Sampling temperature; ``None`` omits the parameter.
        """
        from google import genai  # lazy: only needed when this provider is selected

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Send one request and return the response text.

        Args:
            system: The system prompt (sent as ``system_instruction``).
            user: The rendered transcript.

        Returns:
            The response text.

        Raises:
            LLMError: The request failed or returned no text.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=config,
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return _require(getattr(response, "text", None), "Gemini")


class ScriptedLLM:
    """Replays a fixed list of responses, in order.

    Used by the test suite so every run is deterministic and no test touches the network
    (Rule R9).
    """

    def __init__(self, responses: Sequence[str]) -> None:
        """Store the script.

        Args:
            responses: The replies to return, one per ``complete`` call.
        """
        self._responses = list(responses)
        self._index = 0
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        """Return the next scripted response.

        Args:
            system: Recorded for assertions; otherwise ignored.
            user: Recorded for assertions; otherwise ignored.

        Returns:
            The next reply in the script.

        Raises:
            LLMError: The script ran out, which means the loop iterated more than expected.
        """
        self.calls.append((system, user))
        if self._index >= len(self._responses):
            raise LLMError(
                f"ScriptedLLM exhausted after {len(self._responses)} responses — "
                "the agent asked for one more than the script provides."
            )
        response = self._responses[self._index]
        self._index += 1
        return response


_LOG_LINE = re.compile(r"^Log line:[ \t]*(?P<line>.+)$", re.MULTILINE)
_SEVERITY_OBS = re.compile(r"severity=(?P<severity>\w+)[ \t]+signature=(?P<signature>\S+)")
_INCIDENTS_OBS = re.compile(r"^incidents = (?P<payload>\[.*\])$", re.MULTILINE)
_ACTION_OBS = re.compile(r"action=(?P<action>\w+) \(rule=(?P<rule>\w+)\) — (?P<justification>.+)")


class FakeLLM:
    """A deterministic stand-in that drives a correct ReAct loop on any log line.

    Selected automatically when ``ANTHROPIC_API_KEY`` is unset (Rule R7). It reads the rendered
    transcript, works out which step of the chain has not happened yet, and emits it — so the
    offline demo produces a real four-step trace rather than a canned one.

    It is not a model: it never explores, never errs, and never surprises. It exists so the
    project is runnable and gradeable without a key.
    """

    def __init__(self) -> None:
        """Initialise the call counter."""
        self.call_count = 0

    def complete(self, system: str, user: str) -> str:
        """Emit the next step of the triage chain given what the transcript already shows.

        Args:
            system: Ignored — the fake does not read tool descriptions.
            user: The rendered transcript.

        Returns:
            A well-formed ReAct step.
        """
        self.call_count += 1
        severity_match = _SEVERITY_OBS.search(user)
        incidents_match = _INCIDENTS_OBS.search(user)
        action_match = _ACTION_OBS.search(user)

        if severity_match is None:
            return self._classify(user)
        if incidents_match is None:
            return self._lookup(severity_match)
        if action_match is None:
            return self._recommend(severity_match, incidents_match)
        return self._finish(severity_match, incidents_match, action_match)

    @staticmethod
    def _classify(transcript: str) -> str:
        """First step: nothing is known, so classify the line."""
        match = _LOG_LINE.search(transcript)
        log_line = match.group("line").strip() if match else transcript.strip().splitlines()[0]
        return (
            "Thought: I do not know how severe this line is yet, so classify it first.\n"
            "Action: classify_severity\n"
            f"Action Input: {json.dumps({'log_line': log_line})}"
        )

    @staticmethod
    def _lookup(severity: re.Match[str]) -> str:
        """Second step: severity known, so check whether this has happened before."""
        args = {
            "signature": severity.group("signature"),
            "severity": severity.group("severity"),
        }
        return (
            f"Thought: Severity is {severity.group('severity')}. Check whether this signature "
            "has been seen before.\n"
            "Action: lookup_incidents\n"
            f"Action Input: {json.dumps(args)}"
        )

    @staticmethod
    def _recommend(severity: re.Match[str], incidents: re.Match[str]) -> str:
        """Third step: history known, so apply the escalation policy."""
        args = {
            "severity": severity.group("severity"),
            "incidents": json.loads(incidents.group("payload")),
        }
        return (
            "Thought: I have the severity and the incident history. Apply the escalation "
            "policy rather than deciding myself.\n"
            "Action: recommend_action\n"
            f"Action Input: {json.dumps(args)}"
        )

    @staticmethod
    def _finish(severity: re.Match[str], incidents: re.Match[str], action: re.Match[str]) -> str:
        """Fourth step: everything is known, so conclude."""
        matched: list[str] = [
            str(item.get("incident_id"))
            for item in json.loads(incidents.group("payload"))
            if isinstance(item, dict) and item.get("incident_id")
        ]
        payload: dict[str, Any] = {
            "action": action.group("action"),
            "severity": severity.group("severity"),
            "confidence": 0.86,
            "justification": action.group("justification").strip(),
            "matched_incidents": matched,
        }
        return (
            "Thought: I have a severity, the incident history, and a policy verdict. "
            "That is everything the decision needs.\n"
            f"Final Answer: {json.dumps(payload)}"
        )
