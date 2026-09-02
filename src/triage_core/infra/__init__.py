"""Cross-cutting infrastructure: the shared result cache and trace rendering.

Serves both parts. One cache implementation and one trace renderer, so the two parts produce
comparable output (Rules R3 and R8).
"""

from triage_core.infra.cache import CacheOutcome, CacheStats, ResultCache
from triage_core.infra.tracing import (
    Tracer,
    render_node_trace,
    render_summary,
    render_trace,
    render_transcript,
)

__all__ = [
    "CacheOutcome",
    "CacheStats",
    "ResultCache",
    "Tracer",
    "render_node_trace",
    "render_summary",
    "render_trace",
    "render_transcript",
]
