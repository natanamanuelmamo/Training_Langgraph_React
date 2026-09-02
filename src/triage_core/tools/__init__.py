"""The three triage tools, defined once and shared by both parts (Rule R3).

Serves both parts. Part 1's ``while`` loop and Part 2's ``act`` node dispatch through the same
registry built here — only orchestration differs between them.
"""

from triage_core.tools.base import Tool
from triage_core.tools.classify_severity import ClassifySeverityTool
from triage_core.tools.incident_coercion import coerce_incidents
from triage_core.tools.incident_lookup import IncidentLookupTool, load_incidents
from triage_core.tools.recommend_action import RecommendActionTool, decide_action
from triage_core.tools.registry import build_registry, describe_tools

__all__ = [
    "ClassifySeverityTool",
    "IncidentLookupTool",
    "RecommendActionTool",
    "Tool",
    "build_registry",
    "coerce_incidents",
    "decide_action",
    "describe_tools",
    "load_incidents",
]
