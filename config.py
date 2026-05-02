"""Constants for the NL2SQL agent."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

LLAMA_SERVER = "http://localhost:8080/v1"
MODEL_NAME = "Hermes-2-Pro-Llama-3-8B"
DB_PATH = str(PROJECT_ROOT / "finance.duckdb")

MAX_TRIPS = 5
N_CTX = 4096
CTX_THRESHOLD = 0.9

SYSTEM_PROMPT = """\
You are a financial database agent. Answer questions about customer accounts,
transactions, and loans by generating DuckDB SQL queries.

Tools:
- load_skill(name) — Load a domain skill card (loan-analysis, customer-insights,
  transaction-analysis, account-overview). Do this FIRST for every question.
- run_sql(sql) — Execute SQL against DuckDB. Returns rows or an error.
- schema_check(table) — Look up column names and types for a table.
- explain(text) — Summarize results. Ends the interaction.

Rules:
- Always call load_skill before run_sql.
- On SQL error, call schema_check to verify columns, then retry.
- If 0 rows returned, explain that no data matched.
- For out-of-scope questions, explain your scope honestly.
- Respond with tool calls. No markdown between calls.
"""
