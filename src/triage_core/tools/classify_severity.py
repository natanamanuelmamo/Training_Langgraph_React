"""Step 1 of the triage chain: assign a severity and a stable signature to a log line.

Serves both parts (``task01.md`` §2). Rule-based on keywords and patterns, with an LLM
fallback for lines no rule matches. The fallback arrives by injection as a
:class:`~triage_core.domain.ports.TextCompleter`, so ``tools`` never imports ``llm`` and the
layering in Rule R6 stays one-directional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import Formatter
from typing import Any, ClassVar

from triage_core.domain.errors import ToolInputError
from triage_core.domain.models import LogEntry, Severity, SeverityResult, ToolResult
from triage_core.domain.ports import TextCompleter
from triage_core.domain.prompts import SEVERITY_FALLBACK_SYSTEM
from triage_core.tools.log_parsing import extract_key_values, parse_log_line

_RULE_CONFIDENCE = 0.9
_LEVEL_CONFIDENCE = 0.5
_FALLBACK_CONFIDENCE = 0.6


@dataclass(frozen=True, slots=True)
class _Rule:
    """One row of the classification table."""

    rule_id: str
    pattern: re.Pattern[str]
    signature_template: str
    severity: Severity


#: Ordered — the first match wins, so the most specific patterns come first.
_RULES: tuple[_Rule, ...] = (
    _Rule(
        "db_pool_exhausted",
        re.compile(r"connection pool exhausted|pool exhausted", re.IGNORECASE),
        "db.pool.exhausted:{db}",
        Severity.CRITICAL,
    ),
    _Rule(
        "oom_killed",
        re.compile(r"out of memory|oom[- _]?kill", re.IGNORECASE),
        "oom.killed:{service}",
        Severity.CRITICAL,
    ),
    _Rule(
        "tls_cert_expired",
        re.compile(r"certificate\b.*\bexpired|tls\b.*\bexpired|cert\b.*\bexpired", re.IGNORECASE),
        "tls.cert.expired:{service}",
        Severity.CRITICAL,
    ),
    _Rule(
        "disk_full",
        re.compile(r"no space left|disk full|disk usage at 100", re.IGNORECASE),
        "disk.full:{service}",
        Severity.CRITICAL,
    ),
    _Rule(
        "upstream_timeout",
        re.compile(r"\btimed out\b|\btimeout\b", re.IGNORECASE),
        "net.timeout:{service}",
        Severity.WARNING,
    ),
    _Rule(
        "retry_degraded",
        re.compile(r"\bretrying\b|\bdegraded\b|\bbackoff\b", re.IGNORECASE),
        "svc.degraded:{service}",
        Severity.WARNING,
    ),
    _Rule(
        "deprecation",
        re.compile(r"deprecat", re.IGNORECASE),
        "cfg.deprecated:{service}",
        Severity.INFO,
    ),
    _Rule(
        "cache_warm",
        re.compile(r"cache warm|warming cache|warmed cache", re.IGNORECASE),
        "cache.warm:{service}",
        Severity.INFO,
    ),
)

_LEVEL_DEFAULTS: dict[str, Severity] = {
    "TRACE": Severity.INFO,
    "DEBUG": Severity.INFO,
    "INFO": Severity.INFO,
    "NOTICE": Severity.INFO,
}

_SIGNATURE_SAFE = re.compile(r"[^a-z0-9._:-]+")


def _slug(value: str) -> str:
    """Normalise a placeholder value so signatures stay stable and comparable."""
    return _SIGNATURE_SAFE.sub("-", value.strip().lower()).strip("-") or "unknown"


def _resolve_signature(template: str, entry: LogEntry) -> str:
    """Fill a signature template from the line's ``key=value`` tail, then its service name.

    Args:
        template: A template such as ``db.pool.exhausted:{db}``.
        entry: The parsed log line.

    Returns:
        The template with every placeholder resolved; unresolvable ones become ``unknown``.
    """
    pairs = extract_key_values(entry.message)
    values: dict[str, str] = {}
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is None:
            continue
        raw = pairs.get(field_name) or (entry.service if field_name == "service" else None)
        if raw is None:
            raw = entry.service or "unknown"
        values[field_name] = _slug(raw)
    return template.format(**values)


class ClassifySeverityTool:
    """Classify a log line's severity and derive a stable signature for it."""

    name: ClassVar[str] = "classify_severity"
    description: ClassVar[str] = (
        "Assign a severity (critical/warning/info) to a raw log line and derive the stable "
        "signature that identifies this failure mode. Call this first — nothing else can be "
        "decided until you know how bad the line is."
    )
    input_schema: ClassVar[dict[str, str]] = {"log_line": "string — the raw log line to classify"}

    def __init__(self, llm: TextCompleter) -> None:
        """Store the completer used only when no rule matches.

        Args:
            llm: Injected text completer for the fallback path.
        """
        self._llm = llm

    def run(self, **kwargs: Any) -> ToolResult[SeverityResult]:
        """Classify one log line.

        Args:
            **kwargs: Must contain ``log_line`` as a non-empty string.

        Returns:
            A result carrying the severity, signature, confidence and which rule fired.

        Raises:
            ToolInputError: ``log_line`` is missing, not a string, or empty.
        """
        log_line = kwargs.get("log_line")
        if log_line is None:
            raise ToolInputError("classify_severity requires a 'log_line' argument.")
        if not isinstance(log_line, str):
            raise ToolInputError(f"'log_line' must be a string, got {type(log_line).__name__}.")
        if not log_line.strip():
            raise ToolInputError("'log_line' must not be empty.")
        if unexpected := set(kwargs) - {"log_line"}:
            raise ToolInputError(
                f"classify_severity got unexpected argument(s): {sorted(unexpected)}. "
                "It takes only 'log_line'."
            )

        entry = parse_log_line(log_line)

        for rule in _RULES:
            if rule.pattern.search(entry.message):
                return self._ok(
                    SeverityResult(
                        severity=rule.severity,
                        signature=_resolve_signature(rule.signature_template, entry),
                        confidence=_RULE_CONFIDENCE,
                        matched_rule=rule.rule_id,
                    )
                )

        if entry.level and (severity := _LEVEL_DEFAULTS.get(entry.level)):
            return self._ok(
                SeverityResult(
                    severity=severity,
                    signature=_resolve_signature("log.routine:{service}", entry),
                    confidence=_LEVEL_CONFIDENCE,
                    matched_rule="level_default",
                )
            )

        return self._classify_via_llm(entry)

    def _classify_via_llm(self, entry: LogEntry) -> ToolResult[SeverityResult]:
        """Ask the model when no rule matched and the level is not decisive."""
        raw = self._llm.complete(SEVERITY_FALLBACK_SYSTEM, entry.raw)
        parsed = self._parse_fallback(raw)
        if parsed is None:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=(
                    "no rule matched and the fallback classifier returned an unusable answer; "
                    f"expected 'severity:' and 'signature:' lines, got: {raw.strip()[:160]!r}"
                ),
            )
        severity, signature = parsed
        return self._ok(
            SeverityResult(
                severity=severity,
                signature=signature,
                confidence=_FALLBACK_CONFIDENCE,
                matched_rule="llm_fallback",
            )
        )

    @staticmethod
    def _parse_fallback(raw: str) -> tuple[Severity, str] | None:
        """Read the two-line fallback reply, returning ``None`` if it is unusable."""
        severity: Severity | None = None
        signature: str | None = None
        for line in raw.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "severity":
                try:
                    severity = Severity(value.lower())
                except ValueError:
                    return None
            elif key == "signature" and value:
                signature = _SIGNATURE_SAFE.sub("-", value.lower()).strip("-")
        if severity is None or not signature:
            return None
        return severity, signature

    def _ok(self, result: SeverityResult) -> ToolResult[SeverityResult]:
        """Wrap a successful classification, rendering the Observation text."""
        return ToolResult(
            tool_name=self.name,
            ok=True,
            payload=result,
            detail=(
                f"severity={result.severity} signature={result.signature} "
                f"(rule={result.matched_rule}, confidence={result.confidence:.2f})"
            ),
        )
