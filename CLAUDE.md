# CLAUDE.md

Project instructions for Claude Code. Read this file **before** doing anything in this
repository. It defines the goal, the required architecture, and the rules you must follow.

---

## 1. What this repository is

A training deliverable for **"Agentic Task Design with LangGraph — ReAct & ReWOO Prompting
Patterns"** (Trainer: Kewser Seid, iCog Labs).

The assignment is to solve **one new multi-step problem two different ways**:

| Part | Implementation | Spec |
|------|----------------|------|
| Part 1 | ReAct loop written from scratch in **raw Python** — no LangChain, no LangGraph | `docs/task01.md` |
| Part 2 | The **same problem** as a real **LangGraph** agent, with caching and a human-in-the-loop checkpoint | `docs/task02.md` |

The point of the exercise is to prove understanding of the reason → act → observe → reason
loop *itself*, and then to show that same logic expressed idiomatically in LangGraph.

### Required reading before planning

1. `docs/task01.md` — full Part 1 requirements
2. `docs/task02.md` — full Part 2 requirements
3. `docs/Agentic_Task_Design_with_LangGraph__ReAct_and_ReWOO_Patterns.pdf` — the source
   training deck; it defines the vocabulary and the expected shape of the solution

---

## 2. Chosen problem: Log-Triage Agent

Do **not** use the Athlete Readiness Assistant from the deck — it was fully worked through in
training and tests nothing.

**Log-Triage Agent.** Given a raw error log line, the agent must:

1. Classify the severity of the log (`critical` / `warning` / `info`)
2. Look up past incidents matching that log's signature
3. Decide a final action: `page_on_call`, `file_ticket`, or `ignore`

This satisfies the assignment constraints: **at least two tools**, **a step that depends on a
lookup**, and **a final decision output**.

The sensitive step — the equivalent of "see physio" in the training deck — is
**`page_on_call`**. Paging a human at 3am is the action that must never fire without approval.
That is where the Part 2 HITL checkpoint goes.

---

## 3. Working agreement

### Plan before code

When asked to implement, produce a written implementation plan **first** and stop for approval.
The plan must cover: file tree, module responsibilities, state/data shapes, tool signatures,
control-flow description, and the test list. Do not write implementation code until the plan is
accepted.

### Build order

1. Shared foundation — `domain/`, `tools/`, `llm/`, `infra/`, `data/`
2. Part 1 — `scratch/`
3. Part 2 — `graph/`
4. Tests, README, run scripts

Part 2 must **reuse** the shared tool layer built for Part 1. Duplicating tool logic across the
two parts is a design failure, not a convenience.

---

## 4. Required architecture

The package layout is **three sibling packages**: one framework-free shared core plus two
descriptively-named orchestration packages, one per assignment part. The dependency direction is
`react_from_scratch → triage_core ← react_langgraph`; the two parts never import each other. This
makes Rule R1 a package boundary rather than a naming convention — `langgraph` appears in exactly
one directory.

```
.
├── CLAUDE.md
├── README.md                      # how to install, configure, run both parts
├── pyproject.toml                 # deps, ruff + mypy + pytest config
├── .env.example                   # one provider key (ANTHROPIC / OPENAI / GEMINI / GROQ) + TRIAGE_* knobs
├── .gitignore
│
├── docs/
│   ├── task01.md                  # Part 1 requirements
│   ├── task02.md                  # Part 2 requirements
│   ├── task_1_implementation.md   # Part 1 implementation plan (approved)
│   ├── task_2_implementation.md   # Part 2 implementation plan (approved)
│   ├── graph.md                   # Part 2 — exported mermaid diagram
│   └── Agentic Task Design with LangGraph - ReAct and ReWOO Patterns.pdf
│
├── data/
│   ├── incidents.json             # seed incident history for the lookup tool
│   └── samples/                   # one sample log per decision branch
│
├── src/
│   ├── triage_core/               # SHARED by both parts — framework-free, pure Python
│   │   ├── __init__.py
│   │   ├── config.py              # env loading, settings object, model name, limits
│   │   │
│   │   ├── domain/                # framework-agnostic core — imports only the stdlib
│   │   │   ├── __init__.py
│   │   │   ├── models.py          # LogEntry, Severity, Incident, TriageDecision, ToolResult...
│   │   │   ├── ports.py           # TextCompleter/ToolSpec protocols — keep tools off llm
│   │   │   ├── outcomes.py        # decision building, shared by both parts (parity)
│   │   │   ├── prompts.py         # all prompt text lives here, nowhere else
│   │   │   └── errors.py          # typed exceptions
│   │   │
│   │   ├── tools/                 # SHARED by Part 1 and Part 2 — pure Python
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Tool protocol: name, description, schema, run()
│   │   │   ├── registry.py        # name -> Tool mapping, lookup + describe-for-prompt
│   │   │   ├── classify_severity.py
│   │   │   ├── log_parsing.py     # raw line -> LogEntry, used by classify_severity
│   │   │   ├── incident_lookup.py
│   │   │   ├── incident_coercion.py  # JSON round-trip -> Incident, for recommend_action
│   │   │   └── recommend_action.py   # the escalation policy, and nothing else
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── client.py          # Anthropic/OpenAI/Groq/Gemini wrappers + ScriptedLLM/FakeLLM
│   │   │   └── parsing.py         # strict parse of Thought/Action/Action Input blocks
│   │   │
│   │   └── infra/
│   │       ├── __init__.py
│   │       ├── cache.py           # keyed cache w/ hit/miss stats — used by both parts
│   │       └── tracing.py         # structured step logging / trace rendering
│   │
│   ├── react_from_scratch/        # PART 1 — raw Python ReAct
│   │   ├── __init__.py
│   │   ├── scratchpad.py          # accumulated Thought/Action/Observation transcript
│   │   ├── agent.py               # the while loop — the heart of Part 1
│   │   └── cli.py                 # python -m react_from_scratch.cli "<log line>"
│   │
│   └── react_langgraph/           # PART 2 — LangGraph
│       ├── __init__.py
│       ├── deps.py                # NodeDeps — injected llm/tools/cache/settings
│       ├── state.py               # TriageState TypedDict
│       ├── nodes.py               # reason / act / decide / page_on_call / notify / halt
│       ├── routing.py             # conditional edge functions
│       ├── builder.py             # StateGraph wiring + compile(checkpointer, interrupt_before)
│       ├── hitl.py                # approve / reject / resume helpers
│       └── cli.py                 # python -m react_langgraph.cli "<log line>"
│
└── tests/
    ├── conftest.py                # shared fixtures — never imports LangGraph
    ├── graph_helpers.py           # Part 2 harness, imported only by graph tests
    ├── test_models.py
    ├── test_tools.py
    ├── test_parsing.py
    ├── test_cache.py
    ├── test_react_from_scratch.py
    ├── test_cli.py
    ├── test_graph_routing.py
    ├── test_graph_hitl.py
    ├── test_graph_cli.py
    ├── test_parity.py             # Part 1 and Part 2 agree on the same scripted run
    └── test_isolation.py          # asserts Part 1 imports no langchain/langgraph
```

LangGraph is an **optional** dependency (`pip install -e ".[graph]"`). Part 1 installs and runs
without it, and the graph test modules guard their imports with `pytest.importorskip`, so the
suite is green either way — Rule R1 made checkable rather than merely asserted.

Both parts are runnable end to end from the command line. Neither part is a notebook.

---

## 5. Rules

### R1 — Part 1 is framework-free (hard rule)

Nothing under `src/react_from_scratch/`, and nothing it imports, may import `langchain`,
`langchain_core`, `langgraph`, `llama_index`, or any other agent framework. Because Part 1
imports `triage_core`, this constraint binds `triage_core` too — a single framework import
there invalidates Part 1. `tests/test_isolation.py` enforces this by walking the import graph
both at runtime and statically. If that test fails, Part 1 is invalid.

The ban is on **agent frameworks**, not on plain model SDKs. Alongside `anthropic`, the LLM
SDKs `openai`, `groq`, and `google-genai` are permitted (see §7 / R14) — each is a thin HTTP
client, lazy-imported inside its client class, in the same category as `anthropic`. Do **not**
use `langchain-google-genai` for Gemini: it imports `langchain_core`, which this rule bans and
`test_isolation.py` blocks.

### R2 — The control flow must be visible

In Part 1 the loop is an actual `while` loop and the tools are an actual `dict` of callables.
No hidden dispatch, no clever metaprogramming, no framework doing the sequencing. A reader
should be able to point at the line where "reason" happens and the line where "act" happens.

### R3 — Same problem, same tools, both parts

Part 2 solves the identical problem with the identical tool implementations imported from
`tools/`. Only the orchestration changes. If a bug is fixed in a tool, both parts get the fix.

### R4 — Prompts live in `domain/prompts.py`

No inline prompt strings anywhere else. No f-string prompts buried inside a node or a loop body.

### R5 — Type everything

Full type hints on every function and every public attribute. Domain objects are
`dataclass`es or `TypedDict`s, never bare dicts passed around. `mypy` runs clean.

### R6 — Layering is one-directional

`domain` ← `tools` ← `llm`/`infra` ← `react_from_scratch`/`react_langgraph` ← `cli`. Lower layers
never import higher ones. `domain` imports only the stdlib. `react_from_scratch` and
`react_langgraph` never import each other, and nothing in `triage_core` imports either of them.

Where a lower layer needs a higher one's capability — `classify_severity` needing an LLM fallback —
it depends on a `Protocol` declared in `domain/ports.py` and receives the concrete object by
injection. Protocols are structural, so no import edge is created in either direction.

### R7 — No secrets in the repo

Config comes from environment variables via `config.py`. Ship `.env.example` only. If no
provider key is set (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` /
`GROQ_API_KEY`), the code falls back to the deterministic `FakeLLM` so the whole project is
demonstrable and testable offline. Print a clear one-line warning when that happens.

### R8 — Every run produces a readable trace

Both parts print the full reasoning trace: each iteration's Thought, Action, Action Input, and
Observation, plus a final summary line with iteration count, tool calls, cache hits/misses, and
the final decision. The trace is the evidence that the pattern works — treat it as a
deliverable, not as debug output.

### R9 — Deterministic tests

Tests never call a live API. They use `FakeLLM` with scripted responses and the in-memory
checkpointer. No network in the test suite.

### R10 — Guard the loop

The Part 1 loop and the Part 2 graph both enforce `MAX_ITERATIONS` (default 6) and terminate
cleanly with a partial result and a clear reason rather than spinning or raising. Unknown tool
names and malformed LLM output become an Observation fed back into the loop, not a crash.

### R11 — Small modules, docstring on each

Every module opens with a docstring saying what it is and which part of the assignment it
serves. Keep modules focused; if one passes ~200 lines, split it.

### R12 — Style

`ruff` for lint and format, line length 100, Google-style docstrings, `snake_case` modules.
Run `ruff check . && mypy src && pytest` before declaring anything done.

### R13 — Explain the mapping

`README.md` must contain a short table mapping each assignment requirement to the file and
function that satisfies it, so a reviewer can verify the submission without reading everything.

### R14 — Provider-agnostic core

Every LLM backend implements the `TextCompleter` protocol (`domain/ports.py`):
`complete(system: str, user: str) -> str`, raising `LLMError` on failure. Adding a provider is
one class in `llm/client.py` (SDK lazy-imported in `__init__`), one branch in
`config.build_llm`, and one row in `config.PROVIDER_KEYS` / `DEFAULT_MODELS` — nothing else
changes, because the three `.complete()` call sites never know which backend they hold. The
provider is auto-detected from whichever API key is set; `TRIAGE_PROVIDER` forces a choice.

---

## 6. Definition of done

- [x] `python -m react_from_scratch.cli "<log line>"` runs a full ReAct loop and prints a trace
- [x] `python -m react_langgraph.cli "<log line>"` runs the LangGraph agent and prints a trace
- [x] The graph loops back from `act` to `reason` via a conditional edge
- [x] At least one tool call is cached, with hits visible in the trace
- [x] The graph pauses via `interrupt_before` ahead of the sensitive step and resumes after approval
- [x] Rejecting at the checkpoint changes the outcome — it is a real decision point, not a prompt
- [x] `test_isolation.py` passes, proving Part 1 is framework-free
- [x] `ruff check .`, `mypy src`, and `pytest` all clean
- [x] README documents setup, both run commands, a sample trace, and the requirement mapping

---

## 7. Scope

The assignment text requires **ReAct** in both parts. ReWOO is taught in the deck for
comparison and is **out of scope** unless explicitly requested. If asked to add it later, it
goes in a sibling package `src/rewoo_langgraph/` (planner / worker / solver nodes) reusing the
same `triage_core` tool layer — never by modifying `react_from_scratch/` or `react_langgraph/`.

Do not add dependencies beyond: `anthropic`, `langgraph`, `langchain-core`, `python-dotenv`,
`pytest`, `ruff`, `mypy`, and — in the **optional `providers` extra** — `openai`, `groq`,
`google-genai`. Ask before introducing anything else.

**Deviation on record (multi-provider LLM support).** The assignment is Anthropic-first;
`OpenAILLM` / `GroqLLM` / `GeminiLLM` were added at the user's explicit request so that a key
swap in `.env` changes provider. They live only in `triage_core/llm/client.py`, use their
official SDKs lazy-imported, and are gated behind the optional `providers` extra so the lean
install stays `anthropic`-only. None is an agent framework, so `test_isolation.py` still
passes and R1 still holds. See R14 for the mechanism.
`langchain-google-genai` is **not** used for Gemini — it would pull in `langchain_core`.
Install everything with `pip install -e ".[dev,graph,providers]"`.
