"""Part 1 — the ReAct loop written from scratch in raw Python.

No LangChain, no LangGraph, no agent framework of any kind (Rule R1). The Anthropic SDK is the
only external dependency reachable from this package, and ``tests/test_isolation.py`` proves it
by walking the import graph both at runtime and statically.

Read ``agent.py`` first: it holds the ``while`` loop and the tool ``dict`` that are the point of
this half of the assignment.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
