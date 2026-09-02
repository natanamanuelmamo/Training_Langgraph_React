"""Every prompt string in the project (Rule R4).

Serves both parts. No prompt text lives anywhere else — no inline strings in a tool, a loop
body, or a graph node. The ReAct grammar defined here is the same grammar
``llm.parsing.parse_react_step`` enforces, and ``FINAL_ANSWER_KEYS`` is the single source of
truth shared by the prompt text and the parser's validation.
"""

from __future__ import annotations

from collections.abc import Iterable

from triage_core.domain.ports import ToolSpec

#: Keys the model must supply in its ``Final Answer`` JSON. The parser validates against this
#: exact tuple, and the system prompt below is generated from it, so the two cannot drift.
FINAL_ANSWER_KEYS: tuple[str, ...] = (
    "action",
    "severity",
    "confidence",
    "justification",
    "matched_incidents",
)

VALID_ACTIONS: tuple[str, ...] = ("page_on_call", "file_ticket", "ignore")
VALID_SEVERITIES: tuple[str, ...] = ("critical", "warning", "info")


def render_tool_block(tools: Iterable[ToolSpec]) -> str:
    """Render the available tools for the system prompt.

    Args:
        tools: The tools in the registry, in the order they should be presented.

    Returns:
        A block listing each tool's name, description, and argument schema.
    """
    lines: list[str] = []
    for tool in tools:
        lines.append(f"- {tool.name}")
        lines.append(f"    purpose: {tool.description}")
        args = ", ".join(f"{key} ({desc})" for key, desc in tool.input_schema.items())
        lines.append(f"    arguments: {args or 'none'}")
    return "\n".join(lines)


REACT_SYSTEM_TEMPLATE = """\
You are a Log-Triage Agent for a production software platform. Given one raw error log line,
you decide what an on-call engineer should do about it.

You work in a ReAct loop: you reason about what you still do not know, call one tool, observe
the result, then reason again. You are NOT told the sequence of steps up front — work it out.

AVAILABLE TOOLS
{tool_block}

OUTPUT FORMAT — follow it exactly.

To call a tool, emit exactly:

Thought: <one line on what you still do not know and why this tool answers it>
Action: <one tool name from the list above>
Action Input: <a single JSON object>

To finish, emit exactly:

Thought: <why you now have enough information>
Final Answer: <a single JSON object>

The Final Answer object must contain exactly these keys:
  action             one of: {valid_actions}
  severity           one of: {valid_severities}
  confidence         a number between 0.0 and 1.0
  justification      one or two sentences a human on-call engineer can act on
  matched_incidents  a list of incident id strings, possibly empty

RULES
- Emit ONE Action per turn. Never emit two, and never emit an Action and a Final Answer together.
- Use only the exact tool names listed above. Do not invent tools.
- Action Input must be valid JSON on one line or several — no trailing commas, no comments.
- Do not decide the escalation policy yourself. The `recommend_action` tool owns that rule, so
  call it and adopt its verdict. Your job is choosing which tool to call next.
- Do not guess a severity or an incident history you have not observed. Call the tool.
- If an Observation reports an error, read it, fix the call, and try again.
"""


def build_react_system(tools: Iterable[ToolSpec]) -> str:
    """Build the ReAct system prompt for a given tool registry.

    Args:
        tools: The tools available to the agent this run.

    Returns:
        The fully rendered system prompt.
    """
    return REACT_SYSTEM_TEMPLATE.format(
        tool_block=render_tool_block(tools),
        valid_actions=", ".join(VALID_ACTIONS),
        valid_severities=", ".join(VALID_SEVERITIES),
    )


#: Prepended to the rendered transcript that forms the user turn of every reason step.
SCRATCHPAD_HEADER = "Here is the triage so far. Continue the loop.\n"


SEVERITY_FALLBACK_SYSTEM = """\
You classify a single production log line that a rule-based classifier could not match.

Reply with exactly two lines and nothing else:

severity: <critical|warning|info>
signature: <a short stable dotted identifier for this failure mode>

Guidance:
- critical  user-facing outage, data loss, or a service that cannot serve requests
- warning   degraded behaviour, retries, slow paths — needs attention, not a wake-up
- info      routine or cosmetic; no action required

The signature must be stable across occurrences of the same failure, and specific enough to
distinguish failure modes. Shape it like `area.failure:scope`, for example
`db.pool.exhausted:orders-primary` or `net.timeout:search-api`.
"""
