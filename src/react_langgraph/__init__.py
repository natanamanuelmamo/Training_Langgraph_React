"""Part 2 — the same ReAct agent as a real LangGraph graph.

Same problem, same tools, different orchestration. Every tool, prompt and policy is imported
from ``triage_core``; this package contains orchestration only (``task02.md`` §1). What it adds
over Part 1 is what LangGraph is actually for: a conditional loop-back edge, a checkpointer, a
result cache, and a human-in-the-loop pause before the sensitive step.

``builder.py`` is the only module in the project that imports LangGraph — read it first.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
