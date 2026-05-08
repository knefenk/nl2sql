# NL2SQL Agent — Deep Dive Decision Log

> Exported 2026-05-03. Companion to the line-by-line code walkthrough.

## Project at a Glance

```
LOC:       ~750 Python (agent 287, skills 299, app 142, db_setup 151, config 74)
Stack:     DuckDB + llama.cpp + Hermes-2-Pro 8B + Streamlit
Time:      ~2 weeks part-time
Key claim: Fully local NL2SQL agent with self-correction — no cloud, no API keys
Deploy:    15GB RAM, CPU-only, runs on a laptop
```

---

## DECISION 1: Why an Agentic Pipeline Instead of Single-Pass SQL Generation?

**What it is:** 4-step loop: classify → load_skill → run_sql → [error? retry] → explain.

**Why:**
- Single-pass has no error recovery. With an 8B model, SQL errors happen ~30-40% of the time.
- Without tool-calling, you can't verify column names, check schema, or retry.
- The retry arc (run_sql fails → schema_check → correct → succeed) is the entire interview demo value.
- Alternative: DSPy auto-optimization. Rejected — needs labeled examples, adds training dependency.

**Interview framing:** "I chose an agentic loop because LLMs under 10B make SQL errors frequently. Rather than accepting failure, I built a self-correcting pipeline that demonstrates a core agent design principle: the model doesn't need to be perfect — it needs tools to recover when it's wrong."

---

## DECISION 2: Why Hermes-2-Pro 8B? (Model Selection)

| Model | SQL Quality | Tool-Calling | What Failed |
|-------|------------|--------------|-------------|
| duckdb-nsql:7b | Excellent | **None** | No chat template. Single-shot only. |
| llama-3-sqlcoder-8b | Excellent | **Destroyed** | Echoes system prompt instead of emitting `<tool_call>`. |
| Hermes-2-Pro-8B | Decent | **Purpose-built** (90% eval) | Only model that does both. |

**Key insight:** SQL fine-tuning and tool-calling fine-tuning are adversarial under 10B. You can't have both — unless you bridge the SQL gap with skill cards.

**Why bartowski not NousResearch:** bartowski's Q4_K_M GGUF uses imatrix calibration — important quantization technique for preserving output quality. 4.92GB file.

**Interview framing:** "This is the strongest decision in the project. Three models evaluated, two failed for different reasons, one succeeded. The key insight is that SQL fine-tuning destroys tool-calling in small models — so I bridged the SQL gap architecturally with domain skill cards."

---

## DECISION 3: Why DuckDB?

- **Embedded** — single file, no server process, no port, no auth.
- **Analytical** — columnar engine, fast aggregations (COUNT, SUM, AVG, GROUP BY).
- **Demo-friendly** — `pip install duckdb`, one file `finance.duckdb`, zero setup.
- **LLM-friendly** — PostgreSQL-compatible SQL dialect (the model knows it).
- Alternative: SQLite (too general, no analytical optimizations). Postgres (needs server, overkill).

---

## DECISION 4: Why Native `<tool_call>` Format?

Hermes-2-Pro was literally trained on `<tool_call>` / `<tool_response>` tokens. Using OpenAI's `tools` + `tool_choice` parameters would be asking the model to use a format it wasn't trained for.

**Key lesson:** Match the model's training format — not the API abstraction layer.

---

## DECISION 5: Why Domain Skill Cards?

Each skill card contains only 2-3 relevant tables + 3-5 real example queries. The model adapts examples rather than generating SQL from scratch.

**Why this works:** Under 10B params, the model is better at pattern adaptation than abstract schema reasoning. Give it "SELECT c.name, SUM(l.remaining_balance) AS total_owed FROM loans l JOIN customers c..." and it adapts LIMIT, columns, filters to the user's question.

**Why not full schema?** Exhaustive schema prompts (all 4 tables, all columns) → context bloat → model confused → wrong columns. Skill cards keep context lean and focused.

---

## DECISION 6: Why 4 Skills?

loan-analysis, customer-insights, transaction-analysis, account-overview.

- Each maps to a natural banking query domain
- Minimal table overlap (2-3 tables each instead of all 4)
- Each has 3-5 adaptible example queries
- 4 is the sweet spot: too many = classification errors; too few = each too large

---

## DECISION 7: Why Streamlit?

- Python-native — no JS, no HTML, no separate frontend
- `st.chat_input`, `st.chat_message` for chat-like interface
- `st.dataframe`, `st.code`, `st.expander` built-in for rendering results
- Alternative: Gradio (more ML-focused but less polished chat). FastHTML (too new).

---

## DECISION 8: Why llama.cpp Server?

- **CPU-only inference** — runs on laptop, no GPU needed
- **OpenAI-compatible endpoint** — standard `/v1/chat/completions`
- **GGUF format** — quantized models, 4.92GB fits in RAM
- Alternative: Ollama (easier but less control). vLLM (needs GPU). transformers (slower, more VRAM).

---

## DECISION 9: Why No Frameworks? (No LangChain, No LlamaIndex)

287 lines of agent.py. No hidden abstractions.

- **LangChain** — too many layers for a 4-tool agent. Debugging through LangChain's callback system is painful.
- **LlamaIndex** — query engine abstraction overkill. We're doing direct SQL.
- POC rule: if the agent loop is under 300 lines, frameworks add complexity, not remove it.

---

## DECISION 10: Why Pre-Classification Step?

An extra LLM call before the main agent loop asks: "Which skill domain does this question belong to?"

**Why it's worth the cost:**
- Without it: system prompt contains all 4 skill cards → 2000+ tokens before the user's question → context is half-full before querying → model confused.
- With it: system prompt has only 1 skill card → lean, focused, better SQL.
- Trade-off: +1 LLM call, but saves retries from overloaded context.
- Fallback: if classification fails (model outputs unrecognizable skill), defaults to `customer-insights` — never hangs.

---

## DECISION 11: Why Context Summarization Instead of Truncation?

When tokens cross 90% of 4096, don't just chop messages — compress them.

**Truncation (bad):** Dropping middle messages loses SQL queries, errors, intermediate state → can't recover.
**Summarization (good):** Use the model itself to compress the middle trace. Preserves SQL, errors, results. Keeps system prompt + last 3 turns intact. ~30 lines of Python.

---

## DECISION 12: Why the Brace-Block Scanner?

The tool call parser is more defensive than it looks:

1. Try `<tool_call>...</tool_call>` wrapper (trained format)
2. Fall back to scanning any `{...}` block for `{"name":..., "arguments":...}` (bare JSON)
3. Brace-block regex is key-order agnostic — handles `arguments` before `name`

Pure stdlib: `re` + `json.loads`. No json_repair, no json5.

---

## DECISION 13: Why the Self-Correction Arc?

`run_sql` error → inject DuckDB guardrails → model optionally calls `schema_check(table)` → retry.

**DuckDB guardrails auto-injected on error:**
- No `::` type casts → use `CAST(expr AS type)`
- No `ILIKE` → use `LOWER(col) LIKE`
- No `strftime()` → use `EXTRACT` or `date_trunc`
- Date ranges: `col >= 'YYYY-MM-DD' AND col < 'YYYY-MM-DD'`

**Interview framing:** "The self-correction arc is the heart of the agent. When the model generates incorrect SQL, the system doesn't just retry blindly. It injects syntax rules, the model can verify columns via schema_check, and then regenerates. This transforms ~60% first-pass accuracy into near-100%."

---

## DECISION 14: Why schema_check as a Separate Tool?

- **Opt-in** — model only requests column info when confused
- **Lean** — just column names + types, no example queries
- **Visible** — the recovery arc is explicit in the UI: error → schema_check → fix → success

---

## DECISION 15: Why Synthetic Malaysian Data?

- 50 customers, ~100 accounts, ~2000 transactions, ~30 loans
- Malaysian names (Alice Tan, Bob Lee) and cities (KL, Penang, JB)
- No privacy concerns
- Domain variety: 4 segments, 4 account types, 12 txn categories, 5 loan types
- Date range 2023-2025

---

## DECISION 16: Why `uv`?

- Single tool: `uv sync` = pip install + venv
- `uv init --app` for flat scripts (not package structure)
- Lockfile for reproducibility
- Rust-backed, fast

---

## DECISION 17: Multi-Turn Navigation Hints

After each tool response, the agent appends an explicit "next step" hint:
- After load_skill: "Now call run_sql with your SQL query."
- After run_sql success: "Now call explain to summarize these results."
- After run_sql error: "Fix the query and call run_sql again, or call schema_check."
- After schema_check: "Now fix your query and call run_sql."

Without these hints, the model completes one tool call and stops. The hints enforce pipeline flow.

---

## Decision Map

```
Constraint: FULLY LOCAL (no APIs, no cloud)
    |
    +-> Model: Hermes-2-Pro 8B (only sub-10B with tool-calling)
    |       |
    |       +-> Format: native <tool_call> (matches training)
    |       +-> Server: llama.cpp (CPU-only, GGUF)
    |       +-> Quant: Q4_K_M from bartowski (imatrix calibrated)
    |
    +-> Database: DuckDB (embedded, analytical, single-file)
    |       |
    |       +-> Data: synthetic, Malaysian, 4 tables
    |       +-> Guardrails: DuckDB-specific syntax rules
    |
    +-> Architecture: agentic pipeline (not single-pass)
    |       |
    |       +-> Classify → load_skill → run_sql → [retry?] → explain
    |       +-> Skill cards: domain-specific schema + examples
    |       +-> Self-correction: schema_check + DuckDB rules
    |       +-> Context management: sliding-window summarization
    |
    +-> Parser: two-stage fallback (wrapped + bare JSON)
    |       +-> Brace-block scanner (key-order agnostic)
    |
    +-> Navigation: explicit "next step" hints in tool responses
    |
    +-> UI: Streamlit (chat, no charts, inline trace)
    |
    +-> Dependencies: uv, no frameworks (287 lines total)
```

---

## Interview Q&A Summary

| Question | Key Point | Duration |
|----------|-----------|----------|
| Elevator pitch | 5 sentences: local, 8B, agentic, self-correcting, 287 lines | 60 sec |
| Hardest challenge | Model selection trade-off (SQL vs tool-calling) | 2 min |
| Why not GPT-4? | Fully local constraint, not a limitation | 1 min |
| Self-correction arc | Error → inject rules → schema_check → retry → explain | 2 min |
| Tool design | 4 tools, minimum viable, pipeline-shaped | 1 min |
| Context management | Prevention (skill cards) + remediation (summarization) | 1 min |
| Production plan | Better model, streaming, eval harness, Postgres | 2 min |
| What would you change? | Eval harness, dual-model, training format lesson | 1 min |
| Failure modes | schema_check dead loop → circuit breaker | 1 min |
| Why hire you? | Systems thinking, pragmatic trade-offs, interview-aware design | 1 min |
