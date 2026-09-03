# Log-Triage Agent — ReAct, built two ways

Training deliverable for **"Agentic Task Design with LangGraph — ReAct & ReWOO Prompting
Patterns"**.

One multi-step problem, solved twice: once as a **hand-written ReAct loop in raw Python**, once
as a **LangGraph graph** with caching and a human-in-the-loop checkpoint. The point is to show
the reason → act → observe → reason loop *itself*, then the same logic expressed idiomatically
in LangGraph.

| Part | Implementation | Status |
|---|---|---|
| Part 1 | `src/react_from_scratch/` — a real `while` loop, a real tool `dict`, no framework | **complete** |
| Part 2 | `src/react_langgraph/` — `StateGraph`, conditional loop-back, cache, `interrupt_before` | **complete** |

Both parts import their tools, prompts, parser, policy and cache from `src/triage_core/`. Only
the orchestration differs — that is the whole point of the exercise, and `tests/test_parity.py`
enforces it by running both parts on the same scripted model responses and asserting they reach
the same `TriageDecision`.

---

## The problem: Log-Triage Agent

Given one raw error log line, decide what an on-call engineer should do about it.

```
2026-08-29T03:14:07Z ERROR payments-api  Connection pool exhausted after 30s
    (db=orders-primary, active=100/100, waiting=482)
```

The agent reasons its way through three steps — it is **not** told the sequence:

1. **Classify severity** → `critical` / `warning` / `info`, plus a stable signature
2. **Look up past incidents** matching that signature *(depends on step 1's output)*
3. **Decide an action** → `page_on_call` / `file_ticket` / `ignore`

Output is a `TriageDecision`: action, severity, confidence, justification, matched incident ids.

The **sensitive** step is `page_on_call` — nobody gets woken at 3am without a human saying yes.
That is where Part 2's checkpoint goes.

### Escalation policy

Deterministic, in code (`tools/recommend_action.py`), never in a prompt. The LLM decides *which
tool to call next*; this table decides *what the escalation rule is*. That separation is what
makes the agent auditable. Rules are evaluated top to bottom:

| # | Condition | Action | Rule id |
|---|---|---|---|
| 1 | severity `info` | `ignore` | `info_ignore` |
| 2 | severity `warning` | `file_ticket` | `warning_ticket` |
| 3 | `critical` + no matching incidents | `page_on_call` | `critical_novel` |
| 4 | `critical` + ≥3 recurrences in the lookback window | `page_on_call` | `critical_regression` |
| 5 | `critical` + a prior incident with a known resolution | `file_ticket` | `critical_known_fix` |
| 6 | `critical` + priors, none ever resolved | `page_on_call` | `critical_unresolved` |

Rules 4-before-5 and rule 6 resolve two ambiguities in `docs/task01.md` §2 — reasoning in
`docs/task_1_implementation.md` §11.

---

## Setup

Requires **Python 3.11+**.

```bash
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate               # macOS / Linux

pip install -e ".[dev,graph,providers]"   # both parts + all LLM providers
```

The optional extras keep the core install lean:

```bash
pip install -e ".[dev]"         # Part 1 only — no agent framework, no non-Anthropic SDKs
python -m react_from_scratch.cli --file data/samples/pool_exhausted.log --fake
pytest                          # green; graph + provider test modules skip
```

That is Rule R1 made checkable rather than merely asserted.

### Configuration

```bash
cp .env.example .env            # then fill in ONE provider key
```

**No API key is needed.** With no provider key set, both parts fall back to a deterministic
`FakeLLM` and print a one-line warning. The whole project — including the full test suite —
runs and is gradeable offline (Rule R7).

#### Choosing an LLM provider

The system auto-detects the provider from **whichever API key is present**. Swap the key in
`.env` and the next run uses the new provider — nothing else to change (Rule R14).

| Provider | Key | Default model | SDK (in `[providers]` extra) |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-5` | `anthropic` (always installed) |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | `openai` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` | `google-genai` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | `groq` |

- **Priority** when several keys are set: Anthropic → OpenAI → Gemini → Groq.
- `TRIAGE_PROVIDER=openai|gemini|groq|anthropic` forces a choice regardless of which keys are set.
- Non-Anthropic providers need `pip install -e ".[providers]"`; a clear `LLMError` is raised if
  the SDK is missing.

| Variable | Default | Meaning |
|---|---|---|
| `TRIAGE_PROVIDER` | *(auto)* | Force a provider instead of key-based detection |
| `TRIAGE_MODEL` | *(provider default, see table)* | Model id for the reason step. **Clear or update it when switching providers** — a `claude-*` id sent to Groq will fail |
| `TRIAGE_TEMPERATURE` | `0` | Sampling temperature for OpenAI/Gemini/Groq. Empty (`TRIAGE_TEMPERATURE=`) omits the parameter, which some reasoning models require. Anthropic ignores it |
| `TRIAGE_EFFORT` | `low` | **Anthropic-only** extended-thinking effort; other providers ignore it |
| `TRIAGE_MAX_TOKENS` | `16000` | Output cap |
| `TRIAGE_MAX_ITERATIONS` | `6` | Loop cap (Rule R10) |
| `TRIAGE_LOOKBACK_DAYS` | `90` | Recurrence window for the incident lookup |
| `TRIAGE_TODAY` | `2026-08-30` | "Now" for the lookback. The seed data is a fixed historical snapshot, so anchoring this keeps demos deterministic however long from now they are run |

> Multi-provider support is a deliberate, recorded deviation from the training assignment
> (an Anthropic + LangGraph course). See `CLAUDE.md` §7 and R14.

---

## Running Part 1

```bash
python -m react_from_scratch.cli "2026-08-29T03:14:07Z ERROR payments-api Connection pool exhausted after 30s (db=orders-primary)"
python -m react_from_scratch.cli --file data/samples/pool_exhausted.log
python -m react_from_scratch.cli --file data/samples/cert_expired.log --fake
python -m react_from_scratch.cli --file data/samples/oom_recurring.log --max-iters 4
python -m react_from_scratch.cli --file data/samples/slow_query.log --no-cache
python -m react_from_scratch.cli --file data/samples/cache_warm.log --json
```

Exit codes: `0` complete decision, `1` partial (iteration cap hit), `2` usage or data error.

### Sample logs — one per policy branch

| File | Severity | Rule | Action |
|---|---|---|---|
| `data/samples/pool_exhausted.log` | critical | `critical_known_fix` | `file_ticket` |
| `data/samples/oom_recurring.log` | critical | `critical_regression` | `page_on_call` |
| `data/samples/cert_expired.log` | critical | `critical_novel` | `page_on_call` |
| `data/samples/disk_full_unresolved.log` | critical | `critical_unresolved` | `page_on_call` |
| `data/samples/slow_query.log` | warning | `warning_ticket` | `file_ticket` |
| `data/samples/cache_warm.log` | info | `info_ignore` | `ignore` |

### Sample trace

Real output, captured from `python -m react_from_scratch.cli --file data/samples/cert_expired.log --fake`.
Note the four iterations: the agent discovers the sequence rather than being told it, and each
Observation is what makes the next Thought possible.

```
[info] --fake given — using the deterministic FakeLLM.

── ReAct (raw Python) ────────────────────────────────────
Input: 2026-08-30T09:02:11Z ERROR checkout-gateway  TLS handshake failed: server certificate has expired (host=checkout.internal, notAfter=2026-08-30T00:00:00Z)

[1] Thought : I do not know how severe this line is yet, so classify it first.
    Action  : classify_severity {"log_line": "2026-08-30T09:02:11Z ERROR checkout-gateway  TLS handshake
              failed: server certificate has expired (host=checkout.internal,
              notAfter=2026-08-30T00:00:00Z)"}
    Observ. : severity=critical signature=tls.cert.expired:checkout-gateway (rule=tls_cert_expired,
              confidence=0.90)

[2] Thought : Severity is critical. Check whether this signature has been seen before.
    Action  : lookup_incidents {"severity": "critical", "signature": "tls.cert.expired:checkout-gateway"}
    Observ. [MISS]: 0 matches for tls.cert.expired:checkout-gateway in the last 90 days — novel failure
                    incidents = []

[3] Thought : I have the severity and the incident history. Apply the escalation policy rather than deciding myself.
    Action  : recommend_action {"incidents": [], "severity": "critical"}
    Observ. : action=page_on_call (rule=critical_novel) — Critical failure with no matching prior
              incidents — novel, so nobody has a known fix. Page on-call.

[4] Thought : I have a severity, the incident history, and a policy verdict. That is everything the decision needs.
    Final   : page_on_call

── Result ────────────────────────────────────────────────
Action        : page_on_call
Severity      : critical
Confidence    : 0.86
Justification : Critical failure with no matching prior incidents — novel, so nobody has a known fix. Page
                on-call.
Incidents     : none
Iterations    : 4    Tool calls: 3    LLM calls: 4    Cache: 0 hit / 1 miss
```

This is the `critical_novel` branch: a certificate expiry nobody has seen before, so there is no
prior fix to reference and the policy pages. Run `--file data/samples/pool_exhausted.log` to see
the contrasting case — same severity, but two prior incidents with known resolutions, so it files
a ticket instead of waking anyone.

---

## Running Part 2

```bash
python -m react_langgraph.cli "2026-08-29T03:14:07Z ERROR payments-api Connection pool exhausted (db=orders-primary)"
python -m react_langgraph.cli --file data/samples/cert_expired.log --thread-id inc-001 --auto-approve
python -m react_langgraph.cli --file data/samples/cert_expired.log --thread-id inc-001 --auto-reject --note "renewal in flight"
python -m react_langgraph.cli --file data/samples/pool_exhausted.log --demo-cache
python -m react_langgraph.cli --file data/samples/cert_expired.log --resume inc-001 --auto-approve
python -m react_langgraph.cli --print-graph
```

Exit codes: `0` complete, `1` partial (iteration cap hit), `2` usage or data error. With `--json`
the trace and pause narration go to stderr so stdout stays machine-readable.

### The graph

```
              START → reason ⇄ act          ← act → reason is the ReAct loop-back
                        ↓
                   decide / halt
                        ↓
        page_on_call (⏸ paused before) / notify → END
```

Full exported diagram and a guide to reading it: [`docs/graph.md`](docs/graph.md).

### Sample trace — the paging path, approved

```
⏸  PAUSED before node 'page_on_call' — auto-approving
  Proposed action : page_on_call
  Severity        : critical   confidence 0.86
  Signature       : tls.cert.expired:checkout-gateway
  Matched incident: none
  Why             :
      Critical failure with no matching prior incidents — novel, so nobody has a known
      fix. Page on-call.

── LangGraph ReAct ───────────────────────────────────────
thread_id: inc-001

▸ reason        [1]  Thought: I do not know how severe this line is yet, so classify it first. → act
▸ act           [1]  classify_severity → severity=critical signature=tls.cert.expired:checkout-gateway (rule...
▸ reason        [2]  Thought: Severity is critical. Check whether this signature has been seen before. → act
▸ act           [2]  lookup_incidents  CACHE MISS → 0 matches for tls.cert.expired:checkout-gateway in the last 90 days...
▸ reason        [3]  Thought: I have the severity and the incident history. Apply the escalation policy rather than deciding myself. → act
▸ act           [3]  recommend_action → action=page_on_call (rule=critical_novel) — Critical failure with n...
▸ reason        [4]  Thought: I have a severity, the incident history, and a policy verdict. That is everything the decision needs. → decide
▸ decide        [4]  action=page_on_call  requires_approval=True
▸ page_on_call  [4]  paged: sre-primary

── Result ────────────────────────────────────────────────
Action        : page_on_call
Severity      : critical
Confidence    : 0.86
Justification : Critical failure with no matching prior incidents — novel, so nobody has a known fix. Page
                on-call.
Incidents     : none
Iterations    : 4    Tool calls: 3    LLM calls: 4    Cache: 0 hit / 1 miss
```

Three `act → reason` loop-backs, then the pause. Swap `--auto-approve` for `--auto-reject` on the
same input and the last two lines become:

```
▸ notify        [4]  page rejected by human → file_ticket
...
Action        : file_ticket
Justification : Paging was rejected by a human reviewer (renewal in flight). Filing a ticket
                instead. Original recommendation: Critical failure with no matching prior
                incidents — novel, so nobody has a known fix. Page on-call.
```

`page_on_call` never appears in the trace — the node is not merely skipped at the end, it is never
scheduled. See "Human-in-the-loop" below.

### Caching

`--demo-cache` triages the same log twice in one process against one shared cache:

```
run 1  lookup_incidents  MISS
run 2  lookup_incidents  HIT

cache: 1 hit / 1 miss
```

Add `--no-cache` and it prints `0 hit / 2 miss` instead, with the tool body running both times.
`lookup_incidents` is the cached call: deterministic, repeated, and the direct analogue of the
deck's "athlete checks in twice in a day". The cache is a constructor argument, never a module
global, so every test gets a fresh one.

### Human-in-the-loop

The graph is compiled with `interrupt_before=["page_on_call"]` and a `MemorySaver` checkpointer.
Approve and reject are **deliberately asymmetric**, and that asymmetry is the load-bearing part:

| | Call | Why |
|---|---|---|
| Approve | `update_state(config, {...})` then `invoke(None)` | The destination is unchanged. Re-running the routing edge would re-enter `page_on_call` as a fresh pending task, which `interrupt_before` would pause on again — a pause loop. |
| Reject | `update_state(config, {...}, as_node="decide")` then `invoke(None)` | Attributing the write to `decide` makes LangGraph **re-evaluate** `route_after_decide`, which now sees `approval == "rejected"` and routes to `notify`. Without `as_node` the pending `page_on_call` task survives and resuming pages the engineer anyway — a checkpoint that only *delays* the same result. |

`tests/test_graph_hitl.py` proves the reject path with a spy counter incremented inside
`page_on_call`: after a rejection it must be exactly `0`. Asserting on the final state alone
would not catch a node that ran and was then overwritten.

`warning` and `info` inputs never interrupt — `requires_approval` stays false and they go
straight through `notify` to `END`.

**On `--resume`:** `MemorySaver` lives in one process, so a pause cannot be picked up by a *later*
command. `--resume` therefore demonstrates the real mechanism within one process: it invokes,
prints `app.get_state(config).next == ('page_on_call',)`, then resumes from the checkpoint as a
separate `invoke(None, config)` call. Making the pause survive across commands is a one-line swap
of the checkpointer in `build_checkpointer()` — everything else is independent of which saver is
used, but a durable saver is an extra dependency this project has not taken (`CLAUDE.md` §7).

---

## Tests

```bash
pytest                      # deterministic, offline, no network (Rule R9)
ruff check . && ruff format --check .
mypy src
```

Two test files carry most of the weight of the assignment's claims:

- **`tests/test_isolation.py`** — walks the import graph from `react_from_scratch.agent`, at
  runtime *and* statically over the source, asserting no `langchain*`, `langgraph*` or
  `llama_index*` module is reachable. It also checks the detector itself works (a guard that
  cannot fail is worthless), that the two parts never import each other in either direction, and
  that Part 2's nodes, routers and state import no framework — only `builder.py` wires one.
  If this fails, Part 1 is invalid.
- **`tests/test_parity.py`** — runs both parts on the same scripted responses and asserts an
  identical `TriageDecision`, plus byte-identical prompts on every turn. If someone forks the
  escalation policy into `react_langgraph/`, this goes red.

The graph tests guard their imports with `pytest.importorskip("langgraph")`, so the suite is
green whether or not the optional extra is installed.

---

## Requirement mapping (Rule R13)

So a reviewer can verify the submission without reading everything.

### Assignment requirements

| Requirement | Where |
|---|---|
| A new problem, not the Athlete Readiness Assistant | Log triage — `docs/task01.md` §1 |
| At least two tools | Three: `triage_core/tools/{classify_severity,incident_lookup,recommend_action}.py` |
| A step that depends on a lookup | `IncidentLookupTool.run` — keyed by the signature `classify_severity` produced |
| A final decision output | `TriageDecision` in `triage_core/domain/models.py` |
| **Part 1:** a real `while` loop | `react_from_scratch/agent.py` → `run_react_agent`, the `while iteration < max_iterations` block |
| **Part 1:** a real `dict` of callable tools | `triage_core/tools/registry.py` → `build_registry`, dispatched at `agent.py` `tools.get(...)` |
| **Part 1:** reasoning wired up by hand | `agent.py` `# ---- REASON ----` → `llm.complete(...)` then `parse_react_step(...)` |
| **Part 1:** no framework hiding the control flow | `tests/test_isolation.py` |
| **Part 2:** a real LangGraph graph | `react_langgraph/builder.py` → `build_graph` / `build_app` |
| **Part 2:** conditional routing for the loop-back | `react_langgraph/routing.py`, wired at `builder.py` `graph.add_edge("act", "reason")` |
| **Part 2:** caching on a tool call | `triage_core/infra/cache.py`, applied in `nodes.py` `_invoke`; demo via `--demo-cache` |
| **Part 2:** a HITL checkpoint before the sensitive step | `builder.py` `interrupt_before=["page_on_call"]`; `react_langgraph/hitl.py` |

### Part 1 acceptance checklist (`docs/task01.md` §8)

| Item | Where |
|---|---|
| `while` loop and tool `dict` present and obvious | `react_from_scratch/agent.py` |
| Three tools, structured results, shared with Part 2 | `triage_core/tools/`, all returning `ToolResult` |
| Reasoning parsed by our own parser | `triage_core/llm/parsing.py` → `parse_react_step` |
| Malformed output handled as an Observation | `agent.py`, `except ParseError` → `_observe(...)` → `continue` |
| Scratchpad accumulates and feeds the next step | `react_from_scratch/scratchpad.py` → `Scratchpad.render` |
| Iteration cap enforced with a graceful result | `agent.py` → `triage_core/domain/outcomes.py:partial_decision` |
| CLI runs offline with `FakeLLM` and prints the trace | `react_from_scratch/cli.py`, `triage_core/config.py` → `build_llm` |
| Isolation test proves zero framework dependency | `tests/test_isolation.py` |

### Part 2 acceptance checklist (`docs/task02.md` §9)

| Item | Where |
|---|---|
| Real `StateGraph`, typed state, partial-update nodes | `react_langgraph/state.py`, `nodes.py`, `builder.py` |
| Conditional edge implements the `act → reason` loop-back | `builder.py`; `tests/test_graph_hitl.py::test_the_graph_loops_back_from_act_to_reason` |
| `MemorySaver` wired, `thread_id` respected | `builder.py` → `build_checkpointer`; `hitl.py` → `make_config` |
| Caching on `lookup_incidents`, demoable in one command | `nodes.py` → `_invoke`; `cli.py --demo-cache` |
| `interrupt_before=["page_on_call"]` pauses | `builder.py`; `tests/test_graph_hitl.py::test_the_graph_pauses_before_the_sensitive_node` |
| Approve resumes and executes; reject changes the outcome | `hitl.py` → `approve` / `reject`; `test_rejecting_changes_the_outcome` |
| Non-sensitive paths never interrupt | `routing.py` → `route_after_decide`; `test_non_sensitive_paths_never_interrupt` |
| Zero duplication from Part 1 | `tests/test_parity.py` |
| Graph diagram exported | [`docs/graph.md`](docs/graph.md), via `--print-graph` |

### Project rules (`CLAUDE.md` §5)

| Rule | How it is satisfied |
|---|---|
| R1 framework-free Part 1 | `tests/test_isolation.py`, runtime + static checks |
| R2 visible control flow | `agent.py` — literal `while`, literal `dict`, phase banners |
| R3 same tools both parts | `triage_core/tools/` imported by both; `tests/test_parity.py` |
| R4 prompts in one place | `triage_core/domain/prompts.py` — no prompt string elsewhere |
| R5 typed everything | `mypy --strict`; dataclasses in `domain/models.py` |
| R6 one-directional layering | `domain/ports.py` inverts the one dependency that would break it |
| R7 no secrets | `triage_core/config.py`, `.env.example`, `.gitignore` |
| R8 readable trace | `triage_core/infra/tracing.py` → `render_trace` / `render_summary` |
| R9 deterministic tests | `ScriptedLLM` + `MemorySaver`; no network in the suite |
| R10 guarded loop | `max_iterations` + `domain/outcomes.py` → `partial_decision`, shared by Part 1's loop and Part 2's `halt`; errors become Observations in both |
| R11 small documented modules | Every module opens with a docstring naming which part it serves |
| R12 style gate | `ruff check . && mypy src && pytest` |
| R13 this table | here |
| R14 provider-agnostic core | `triage_core/domain/ports.py` → `TextCompleter`; `config.py` → `PROVIDER_KEYS` / `_detect_provider` / `build_llm`; `llm/client.py` → one class per provider |

---

## Layout

```
src/
  triage_core/            shared, framework-free — imported by both parts
    domain/               models, ports, prompts, errors  (stdlib only)
    tools/                the three tools + the registry dict
                          recommend_action.py holds the escalation policy and nothing else;
                          log_parsing.py and incident_coercion.py take the adapter work
    llm/                  Anthropic / OpenAI / Groq / Gemini clients, ScriptedLLM, FakeLLM, ReAct parser
    infra/                result cache, trace rendering
    config.py             settings + provider auto-detection + LLM selection
  react_from_scratch/     PART 1 — scratchpad, THE while loop, CLI
  react_langgraph/        PART 2 — state, nodes, routers, builder, HITL, CLI
                          builder.py is the ONLY module that imports LangGraph
```

Dependency direction is `react_from_scratch → triage_core ← react_langgraph`. The two parts
never import each other, and `triage_core` never imports an agent framework — Part 1 imports it,
so a framework import there would invalidate Part 1.

Full design notes: [`docs/task_1_implementation.md`](docs/task_1_implementation.md) and
[`docs/task_2_implementation.md`](docs/task_2_implementation.md).
