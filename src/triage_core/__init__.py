"""Shared, framework-free core for the Log-Triage Agent.

Serves both parts of the assignment. Part 1 (``react_from_scratch``) and Part 2
(``react_langgraph``) both import their tools, prompts, parser, cache and models from here —
only orchestration differs between them (Rule R3).

This package must never import ``langgraph``, ``langchain`` or any other agent framework:
Part 1 imports it, so a framework import here would invalidate Part 1 (Rule R1).
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
