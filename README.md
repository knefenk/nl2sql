# NL2SQL Agent

Agentic natural-language-to-SQL system using DuckDB and Hermes-2-Pro 8B running locally via llama.cpp.

## Flow

```
USER QUESTION
    |
    v
+------- CLASSIFY ----------+
| Hermes-2-Pro classifies    |     load_skill("loan-analysis")
| the question domain from    | -----------------------------------> injects skill card
| 4 available skills          |                                       (schema + examples)
+-----------------------------+
    |
    v
+------- RUN_SQL -------------+    SELECT c.name, SUM(l.remaining_balance) ...
| Model generates DuckDB SQL   | -----------------------------------> execute against DuckDB
| adapting skill card examples |
+------------------------------+
    |
    |--- SUCCESS ---> +---- EXPLAIN ------+
    |                 | NL summary of     | ---> DISPLAY IN CHAT UI
    |                 | query results     |
    |                 +-------------------+
    |
    |--- ERROR -----> +---- RETRY --------+
                      | Inject DuckDB       |
                      | guardrails.         |
                      | Optionally call     |
                      | schema_check(table) |
                      | to verify columns.  |
                      +---------------------+
                              |
                              v
                         RUN_SQL again
```

## Architecture

**Model:** Hermes-2-Pro-Llama-3-8B (NousResearch), Q4_K_M GGUF (~4.9GB). Chosen for its purpose-built function calling capability (90% accuracy on internal eval) paired with strong general reasoning from the Llama 3 8B base.

**Tool format:** Hermes-2-Pro uses native `<tool_call>` tags, not OpenAI function-calling JSON. This matches the model's training format for reliable tool dispatch:

```
<tool_call>
{"arguments": {"skill": "loan-analysis"}, "name": "load_skill"}
</tool_call>
```

Agent responds with `<tool_response>` blocks containing results. Pipeline navigation is guided by explicit "next step" hints embedded in each response.

**Self-correction:** On SQL error, the agent automatically injects DuckDB guardrails (no `::` casts, date range patterns, `strftime` syntax) and can call `schema_check(table)` to verify column names before retrying.

**Context management:** Sliding-window summarization compresses the agent trace when conversation exceeds 90% of the 4096-token context window, keeping the system prompt and recent turns intact.

## Skills

Four domain skill cards. The model selects one based on the question domain:

| Skill | Tables | Covers |
|-------|--------|--------|
| `loan-analysis` | loans, customers | Interest rates, remaining balances, loan types, date ranges |
| `customer-insights` | customers, accounts | Segments, demographics, balance per customer, join dates |
| `transaction-analysis` | transactions, accounts, customers | Spending categories, date ranges, amounts, account links |
| `account-overview` | accounts, customers | Account types, currencies, balance thresholds |

Each skill card contains only the relevant schema subset and real example queries (not abstract templates). The model adapts these examples to variations in the user's question — e.g., "top 5 by balance" becomes "top 10 by loan amount" by adjusting LIMIT and column references.

```
User: "Top 5 customers by loan balance"
  -> load_skill("loan-analysis")
  -> sees example: "Top N customers by remaining loan balance"
  -> adapts: LIMIT 5, keeps JOIN, uses remaining_balance
```

## Self-Correction Arc

```
run_sql("SELECT customer_name, remaining_amount FROM loans...")
  -> DuckDB: "Column 'remaining_amount' not found"
  -> Agent injects DuckDB rules
  -> Model calls schema_check("loans")
  -> Returns: loan_id, customer_id, loan_type, principal,
             interest_rate, remaining_balance, start_date, term_months
  -> Model corrects: remaining_amount -> remaining_balance
  -> run_sql succeeds
```

## Setup

```bash
# Install dependencies
uv sync

# Generate the sample database (2023-2025 data range)
uv run python db_setup.py

# Start llama.cpp server with Hermes-2-Pro
python -m llama_cpp.server \
  --model Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf \
  --port 8080 \
  --n_ctx 4096

# Launch the chat UI
uv run streamlit run app.py
```

## Project Structure

```
├── config.py      # Server endpoint, model, system prompt, context limits
├── skills.py      # 4 domain skill cards, DuckDB rules, schema lookup
├── agent.py       # Agent loop: classify, tool dispatch, self-correction, context summarization
├── db_setup.py    # Sample database generator (2023-2025, 4 tables)
├── app.py         # Streamlit chat UI with chronological tool-call trace
└── finance.duckdb # Generated database
```

No LangChain, no heavy frameworks. The entire agent loop is ~260 lines of Python with no hidden abstractions.

## Requirements

- Python 3.11+
- llama.cpp with OpenAI-compatible server
- Hermes-2-Pro-Llama-3-8B GGUF (Q4_K_M, ~4.9GB)
- 15GB RAM (CPU-only inference)
