# NL2SQL Agent

Natural-language-to-SQL agent using DuckDB and Qwen 3.5 9B (DeepSeek V4 distilled) running locally via llama.cpp. Fully local, no cloud APIs, no API keys.

## Flow

```
USER QUESTION
    |
    v
+---- LOAD_SKILL ------------+
| Model calls load_skill()    | --> injects skill card (schema + example queries)
| to get domain context       |
+-----------------------------+
    |
    v
+---- RUN_SQL ----------------+   SELECT c.name, SUM(l.remaining_balance) ...
| Model generates DuckDB SQL   | --> executed against DuckDB
| adapting skill card examples |
+------------------------------+
    |
    |--- SUCCESS ---> +---- EXPLAIN ------+
    |                 | NL summary of     | --> DISPLAY IN CHAT UI
    |                 | query results     |
    |                 +-------------------+
    |
    |--- ERROR -----> +---- RETRY --------+
                      | Inject DuckDB       |
                      | guardrails + call   |
                      | schema_check() to   |
                      | verify columns       |
                      +---------------------+
                              |
                              v
                         RUN_SQL again
```

## Architecture

**Model:** Qwen 3.5 9B distilled from DeepSeek V4 Flash. Q4_K_M GGUF (5.4GB, Apache 2.0). Uses native OpenAI function-calling via llama.cpp's chat template — no regex parsing, no text-based tool format.

**Tool dispatch:** Server handles all function-calling formatting natively via the chat template. Tools are defined as standard OpenAI function schemas in `agent.py`. The model calls them via `tool_calls` in the API response.

**Self-correction:** On SQL error, the agent auto-injects DuckDB guardrails and can call `schema_check(table)` to verify column names before retrying.

**Multi-turn context:** Up to 10 previous Q&A pairs are stored with question, skill used, results summary, and answer. Follow-up questions with pronouns ("them", "those", "her") resolve to previous queries by reusing WHERE clauses from context.

**Context window:** 8192 tokens. Sliding-window summarization compresses agent trace when exceeding 90% of context.

## Skills

Four domain skill cards. The model selects one based on the question domain:

| Skill | Tables | Covers |
|-------|--------|--------|
| `loan-analysis` | loans, customers | Interest rates, remaining balances, loan types, date ranges |
| `customer-insights` | customers, accounts | Segments, demographics, balance per customer, join dates |
| `transaction-analysis` | transactions, accounts, customers | Spending categories, date ranges, amounts, account links |
| `account-overview` | accounts, customers | Account types, currencies, balance thresholds |

Each skill card contains only the relevant schema subset and real example queries. The model adapts these examples rather than generating SQL from scratch.

## Setup

```bash
# Install dependencies
uv sync

# Generate the sample database
uv run python db_setup.py

# Start llama.cpp server
llama-server -hf Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF:Q4_K_M \
    --port 8080 -c 8192 --alias qwen3.5-9b-dsv4

# Launch the chat UI
uv run streamlit run app.py
```

## Project Structure

```
├── config.py           # Server endpoint, model, system prompt, context limits
├── skills.py           # 4 domain skill cards, DuckDB rules, schema lookup
├── agent.py            # Agent loop with native function calling + self-correction
├── db_setup.py         # Sample database generator (2023-2025, 4 tables)
├── app.py              # Streamlit chat UI with chronological tool-call trace
├── benchmark_model.py  # SQL quality + tool-calling benchmark suite
└── finance.duckdb      # Generated database
```

No LangChain, no heavy frameworks. ~260 lines for the entire agent loop.

## Requirements

- Python 3.11+
- llama.cpp with OpenAI-compatible server
- Qwen 3.5 9B DeepSeek V4 Flash GGUF (Q4_K_M, ~5.4GB)
- 8GB+ VRAM (GPU) or 16GB RAM (CPU)
