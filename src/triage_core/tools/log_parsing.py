"""Turn a raw log line into a structured :class:`LogEntry`.

Serves both parts, via ``classify_severity``. Split out of ``classify_severity.py`` to keep
both modules small and focused (Rule R11). Pure and tolerant: a line that matches nothing
still yields a usable ``LogEntry`` with ``message`` set to the whole line.
"""

from __future__ import annotations

import re
from datetime import datetime

from triage_core.domain.models import LogEntry

_TIMESTAMP = re.compile(
    r"^\s*(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s*"
)
_LEVEL = re.compile(
    r"^\s*(?:\[)?(?P<level>TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL)(?:\])?\s+",
    re.IGNORECASE,
)
_SERVICE = re.compile(r"^\s*(?P<service>[a-z0-9][a-z0-9._-]{2,})\s+", re.IGNORECASE)

#: ``db=orders-primary``, ``active=100/100`` — the structured tail most log lines carry.
_KEY_VALUE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s,;)\]]+)")

#: Words that look like a service token but are really the start of the message.
_NOT_A_SERVICE = frozenset(
    {"connection", "failed", "unable", "cannot", "could", "the", "out", "no", "disk", "cache"}
)


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z``."""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_log_line(line: str) -> LogEntry:
    """Split a log line into timestamp, level, service and message.

    Every component is optional. Anything that cannot be identified is left as ``None`` and
    the remainder becomes ``message``, so an unrecognised format degrades to "the whole line
    is the message" rather than failing.

    Args:
        line: The raw log line.

    Returns:
        The parsed entry, with ``raw`` preserving the original text.
    """
    rest = line.strip()
    timestamp: datetime | None = None
    level: str | None = None
    service: str | None = None

    if match := _TIMESTAMP.match(rest):
        timestamp = _parse_timestamp(match.group("ts"))
        rest = rest[match.end() :]

    if match := _LEVEL.match(rest):
        level = match.group("level").upper()
        rest = rest[match.end() :]

    if match := _SERVICE.match(rest):
        candidate = match.group("service")
        # A service token is only plausible once we have seen a level, and must not be an
        # ordinary English word that happens to start the message.
        if level is not None and candidate.lower() not in _NOT_A_SERVICE:
            service = candidate
            rest = rest[match.end() :]

    return LogEntry(
        raw=line,
        timestamp=timestamp,
        level=level,
        service=service,
        message=rest.strip() or line.strip(),
    )


def extract_key_values(text: str) -> dict[str, str]:
    """Pull ``key=value`` pairs out of a log line's structured tail.

    These supply the placeholders in a signature template — ``db.pool.exhausted:{db}`` reads
    ``db`` from here.

    Args:
        text: The log message.

    Returns:
        Lower-cased keys mapped to their raw values, first occurrence winning.
    """
    found: dict[str, str] = {}
    for match in _KEY_VALUE.finditer(text):
        key = match.group("key").lower()
        found.setdefault(key, match.group("value"))
    return found
