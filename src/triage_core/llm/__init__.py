"""The reasoning step: model clients and the ReAct output parser.

Serves both parts. No agent framework is imported here. The Anthropic SDK is the only
required external dependency, imported lazily inside ``AnthropicLLM`` (Rule R1). The
``OpenAILLM`` / ``GroqLLM`` / ``GeminiLLM`` backends are optional (``pip install -e
".[providers]"``) and lazy-import their SDKs the same way (Rule R14).
"""

from triage_core.llm.client import (
    AnthropicLLM,
    FakeLLM,
    GeminiLLM,
    GroqLLM,
    LLMError,
    OpenAILLM,
    ScriptedLLM,
)
from triage_core.llm.parsing import parse_react_step

__all__ = [
    "AnthropicLLM",
    "FakeLLM",
    "GeminiLLM",
    "GroqLLM",
    "LLMError",
    "OpenAILLM",
    "ScriptedLLM",
    "parse_react_step",
]
