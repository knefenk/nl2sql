# NL2SQL Agent

Agentic natural-language-to-SQL system using DuckDB and llama-3-sqlcoder-8b running locally via llama.cpp.

## Architecture

```
User question
  -> Classify (which domain skill?)
  -> Load skill card (schema subset + example queries)
  -> Generate SQL
  -> Execute against DuckDB
  -> [Error? -> Inject DuckDB rules, schema_check -> Retry]
  -> Explain results in natural language
```

The agent uses OpenAI-compatible function calling with four tools: `load_skill`, `run_sql`, `schema_check`, and `explain`. Self-correction handles column-not-found errors by looking up actual schema.

## Skills

Four domain skill cards provide schema context and example queries:

| Skill | Domain |
|-------|--------|
| `loan-analysis` | Loans, interest rates, balances |
| `customer-insights` | Customer demographics, segments |
| `transaction-analysis` | Spending, categories, dates |
| `account-overview` | Account balances, types, currencies |

Each skill includes only the relevant tables and real example queries the model adapts.

## Setup

```bash
# Install dependencies
uv sync

# Generate the sample database (50 customers, 104 accounts, 2752 transactions, 51 loans)
uv run python db_setup.py

# Start llama.cpp server with sqlcoder model
python -m llama_cpp.server \
  --model path/to/llama-3-sqlcoder-8b-Q4_K_M.gguf \
  --port 8080 \
  --n_ctx 4096

# Launch the UI
uv run streamlit run app.py
```

## Project Structure

```
nl2sql-agent/
├── config.py      # Server endpoint, model name, system prompt
├── skills.py      # Domain skill cards and schema lookup
├── tools.py        # OpenAI function-calling tool definitions
├── agent.py        # Agent loop with self-correction
├── db_setup.py     # Sample database generator
├── app.py          # Streamlit UI
└── finance.duckdb  # Generated database
```

## Requirements

- Python 3.11+
- llama.cpp with OpenAI-compatible server
- llama-3-sqlcoder-8b (GGUF, Q4_K_M recommended)
- 15GB RAM (CPU-only inference)
