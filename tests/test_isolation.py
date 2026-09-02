"""Rule R1: Part 1 must be framework-free. If this fails, Part 1 is invalid.

Two independent checks, because either alone has a blind spot:

1. **Runtime** — import ``react_from_scratch.agent`` in a clean subprocess and inspect
   ``sys.modules``. Catches transitive imports pulled in by a dependency.
2. **Static** — walk every ``import`` statement in the source of ``react_from_scratch`` *and*
   ``triage_core``. Catches a banned import hidden behind a conditional that did not execute.

``triage_core`` is in scope because Part 1 imports it: a framework import there would be just
as fatal.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

BANNED_PREFIXES = ("langchain", "langgraph", "llama_index", "haystack", "crewai", "autogen")

SRC = Path(__file__).resolve().parents[1] / "src"
GUARDED_PACKAGES = ("react_from_scratch", "triage_core")


def _is_banned(module_name: str) -> bool:
    """True when a module name is, or sits under, a banned top-level package."""
    root = module_name.split(".", 1)[0]
    return root.startswith(BANNED_PREFIXES)


def test_no_framework_is_imported_at_runtime() -> None:
    """Import the agent in a clean interpreter and check what actually loaded."""
    probe = (
        "import json, sys; "
        "import react_from_scratch.agent; "
        "import react_from_scratch.cli; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"probe failed:\n{completed.stderr}"

    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    offenders = sorted({name for name in loaded if _is_banned(name)})
    assert not offenders, (
        f"Part 1 reached an agent framework at runtime: {offenders}. "
        "Rule R1 makes this fatal — Part 1 must be framework-free."
    )


def _source_files() -> list[Path]:
    """Every ``.py`` file in the packages Part 1 depends on."""
    files: list[Path] = []
    for package in GUARDED_PACKAGES:
        files.extend(sorted((SRC / package).rglob("*.py")))
    return files


def test_source_files_were_found() -> None:
    """Guard against the static check silently passing over an empty file list."""
    files = _source_files()
    assert len(files) > 10, f"expected the full source tree, found {len(files)} files"


def _banned_imports_in(source: str, filename: str = "<test>") -> list[str]:
    """Collect every banned module imported anywhere in a source file.

    Walks the whole tree rather than just the top level, so an import tucked inside a function
    or a conditional is caught too.
    """
    tree = ast.parse(source, filename=filename)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(alias.name for alias in node.names if _is_banned(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module and _is_banned(node.module):
            offenders.append(node.module)
    return offenders


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import langgraph", ["langgraph"]),
        ("import langgraph.graph", ["langgraph.graph"]),
        ("from langchain_core.prompts import PromptTemplate", ["langchain_core.prompts"]),
        ("from langchain.agents import initialize_agent", ["langchain.agents"]),
        ("import llama_index", ["llama_index"]),
        ("def f():\n    import langgraph  # hidden inside a function", ["langgraph"]),
        ("if False:\n    from langgraph.graph import StateGraph", ["langgraph.graph"]),
        ("import json\nfrom anthropic import Anthropic", []),
        ("from triage_core.tools.registry import build_registry", []),
        # Deliberately over-broad: prefix matching is required to catch the whole langchain
        # family (langchain_core, langchain_community, ...), so a same-prefixed name is
        # flagged too. Erring toward a false positive is right for a guard whose other
        # failure mode is silently admitting a framework.
        ("import langgraphviz_lookalike", ["langgraphviz_lookalike"]),
    ],
)
def test_the_detector_itself_works(source: str, expected: list[str]) -> None:
    """A guard that cannot fail is worthless — prove this one detects violations."""
    assert _banned_imports_in(source) == expected


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_framework_import_statement_in_source(path: Path) -> None:
    """Statically walk every import statement, including ones inside functions."""
    offenders = _banned_imports_in(path.read_text(encoding="utf-8"), str(path))
    assert not offenders, (
        f"{path.relative_to(SRC)} imports an agent framework: {offenders}. "
        "Rule R1 makes this fatal — Part 1 must be framework-free."
    )


def _imported_names(path: Path) -> list[str]:
    """Every module name imported anywhere in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("react_from_scratch", "react_langgraph"),
        ("react_langgraph", "react_from_scratch"),
    ],
)
def test_part_1_and_part_2_never_import_each_other(package: str, forbidden: str) -> None:
    """Rule R6: the two orchestration packages stay independent, in both directions."""
    root = SRC / package
    if not root.exists():  # pragma: no cover - Part 2 may not be built yet
        pytest.skip(f"{package} is not present")
    for path in sorted(root.rglob("*.py")):
        for name in _imported_names(path):
            assert not name.startswith(forbidden), (
                f"{path.relative_to(SRC)} imports {name} — the two parts must stay separate."
            )


#: Part 2's substance — the nodes, the routers, the state and their dependencies — must be
#: ordinary Python. Only ``builder.py`` (which wires the graph) and ``hitl.py`` and ``cli.py``
#: (which need the compiled-app type) may name the framework. Keeping the split explicit is
#: what shows that the ReAct logic is the same logic Part 1 runs, merely orchestrated
#: differently.
FRAMEWORK_FREE_PART2_MODULES = ("nodes.py", "routing.py", "state.py", "deps.py", "__init__.py")


@pytest.mark.parametrize("module", FRAMEWORK_FREE_PART2_MODULES)
def test_part_2_logic_modules_are_framework_free(module: str) -> None:
    """The graph's nodes, routing and state import no framework of their own."""
    path = SRC / "react_langgraph" / module
    if not path.exists():  # pragma: no cover - Part 2 may not be built yet
        pytest.skip(f"{module} is not present")

    offenders = [name for name in _imported_names(path) if _is_banned(name)]
    assert not offenders, (
        f"react_langgraph/{module} imports {offenders}. Only builder.py should wire the "
        "framework — the nodes, routers and state are meant to be plain Python."
    )


def test_domain_layer_imports_only_the_standard_library() -> None:
    """Rule R6: ``domain`` is the bottom of the graph."""
    allowed_internal = "triage_core.domain"
    for path in sorted((SRC / "triage_core" / "domain").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("triage_core"):
                    assert name.startswith(allowed_internal), (
                        f"{path.relative_to(SRC)} imports {name}; domain may only import "
                        "from within domain."
                    )
                assert not name.startswith(("react_from_scratch", "react_langgraph")), (
                    f"{path.relative_to(SRC)} imports an orchestration package — "
                    "layering is one-directional."
                )
