"""A keyed result cache with hit/miss stats.

Serves both parts (``task02.md`` §4). One implementation, used by Part 1's loop and Part 2's
``act`` node, so a reviewer sees the same counters in both traces (Rule R8).

Injected as a dependency, never a module-level global, so every test gets a fresh instance.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheStats:
    """Running hit/miss tallies for one cache instance."""

    hits: int = 0
    misses: int = 0

    def render(self) -> str:
        """Render as the summary-line fragment, e.g. ``1 hit / 2 miss``."""
        return f"{self.hits} hit / {self.misses} miss"


@dataclass(frozen=True, slots=True)
class CacheOutcome(Generic[T]):
    """One lookup's result, plus how it was served and how long it took.

    ``elapsed_ms`` exists so Part 2's ``--demo-cache`` can show the difference a hit makes
    (``task02.md`` §4). Part 1 ignores it.
    """

    value: T
    hit: bool
    elapsed_ms: float

    @property
    def verdict(self) -> str:
        """``HIT`` or ``MISS``, for the trace."""
        return "HIT" if self.hit else "MISS"


class ResultCache:
    """Memoises tool results by name and arguments.

    Disabling it still counts misses, so ``--no-cache`` produces a visibly different summary
    line rather than silence.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        """Create an empty cache.

        Args:
            enabled: When False, every lookup is a miss and nothing is stored.
        """
        self._enabled = enabled
        self._entries: dict[str, Any] = {}
        self._stats = CacheStats()

    @property
    def enabled(self) -> bool:
        """Whether this cache stores and serves entries."""
        return self._enabled

    @property
    def stats(self) -> CacheStats:
        """The running hit/miss tallies."""
        return self._stats

    @staticmethod
    def key(tool_name: str, kwargs: Mapping[str, Any]) -> str:
        """Build a stable key from a tool name and its arguments.

        ``sort_keys`` makes the key independent of argument order; ``default=str`` handles
        enums and dates that JSON cannot encode directly.

        Args:
            tool_name: The tool being called.
            kwargs: Its arguments.

        Returns:
            A key of the form ``<tool_name>:<16 hex chars>``.
        """
        blob = json.dumps(dict(kwargs), sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return f"{tool_name}:{digest}"

    def get_or_compute(
        self,
        tool_name: str,
        kwargs: Mapping[str, Any],
        compute: Callable[[], T],
    ) -> CacheOutcome[T]:
        """Return a cached value, or compute and store one.

        Args:
            tool_name: The tool being called.
            kwargs: Its arguments, which form part of the key.
            compute: Zero-argument callable producing the value on a miss. Exceptions
                propagate to the caller and nothing is stored.

        Returns:
            The value, whether it was a hit, and how long the lookup took.
        """
        cache_key = self.key(tool_name, kwargs)
        started = time.perf_counter()

        if self._enabled and cache_key in self._entries:
            self._stats.hits += 1
            value: T = self._entries[cache_key]
            return CacheOutcome(value=value, hit=True, elapsed_ms=_elapsed_ms(started))

        self._stats.misses += 1
        computed = compute()
        if self._enabled:
            self._entries[cache_key] = computed
        return CacheOutcome(value=computed, hit=False, elapsed_ms=_elapsed_ms(started))

    def clear(self) -> None:
        """Drop every entry, keeping the stats."""
        self._entries.clear()


def _elapsed_ms(started: float) -> float:
    """Milliseconds since ``started``."""
    return (time.perf_counter() - started) * 1000.0
