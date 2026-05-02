# NL2SQL Agent

Agentic natural-language-to-SQL system using DuckDB and Hermes-2-Pro 8B running locally via llama.cpp.

## Flow

```
USER QUESTION
    |
    v
+------- CLASSIFY -------+      load_skill("loan-analysis")
| Hermes-2-Pro classifies | ------------------------------------> injects skill card
| the question domain      |                                      (schema + examples)
+--------------------------+
    |
    v
+------- RUN_SQL ---------+      SELECT c.name, SUM(l.remaining_balance) ...
| Model generates SQL      | ------------------------------------> DuckDB execution
| from skill card examples |
+--------------------------+
    |
    |--- SUCCESS ---> +---- EXPLAIN ----+
    |                 | NL summary of   | ---> DISPLAY TO USER
    |                 | query results   |
    |                 +-----------------+
    |
    |--- ERROR -----> +---- RETRY ------+
                      | Inject duckdb    |
                      | guardrails.      |
                      | Optionally call  |
                      | schema_check()   |
                      | to verify cols.  |
                      +-----------------+
                              |
                              v
                         RUN_SQL again
```

## Self-Correction

On SQL error, the agent automatically:

1. Injects DuckDB-specific rules (no `::` casts, date range patterns, `strftime` syntax)
2. If column-not-found: calls `schema_check(table)` to look up real column names
3. Retries SQL generation with corrected context

## Skills

Four domain skill cards. The model loads one based on the question:

| Skill | Tables | Covers |
|-------|--------|--------|
| `loan-analysis` | loans, customers | Interest rates, remaining balances, loan types |
| `customer-insights` | customers, accounts | Segments, demographics, balances per customer |
| `transaction-analysis` | transactions, accounts, customers | Spending categories, date ranges, amounts |
| `account-overview` | accounts, customers | Account types, currencies, balance thresholds |

Each skill card contains only relevant schema and real example queries (not abstract templates). The model adapts these examples to user variations (e.g., "top 5" -> "top 10", "by balance" -> "by loan amount").

## Tool Format

Hermes-2-Pro uses native `<tool_call>` format, not OpenAI function-calling JSON:

```
<tool_call>
{"arguments": {"skill": "loan-analysis"}, "name": "load_skill"}
</tool_call>
```

Agent responds with results in `<tool_response>` blocks. Pipeline is guided by explicit "next step" hints after each response.

## Setup

```bash
# Install dependencies
uv sync

# Generate the sample database (50 customers, ~100 accounts, ~2750 transactions, ~50 loans)
uv run python db_setup.py

# Start llama.cpp server with Hermes-2-Pro
python -m llama_cpp.server \
  --model Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf \
  --port 8080 \
  --n_ctx 4096

# Launch the UI
uv run streamlit run app.py
```

## Project Structure

```
├── config.py      # Server endpoint, model name, system prompt, context limits
├── skills.py      # Domain skill cards, DuckDB rules, schema lookup
├── agent.py       # Agent loop: classify, tool dispatch, self-correction, context summarization
├── db_setup.py    # Sample database generator (4 tables, realistic financial data)
├── app.py         # Streamlit chat UI with chronological tool-call trace
└── finance.duckdb # Generated database
```

## Requirements

- Python 3.11+
- llama.cpp with OpenAI-compatible server
- Hermes-2-Pro-Llama-3-8B (GGUF, Q4_K_M, ~4.9GB)
- 15GB RAM (CPU-only inference)
