"""Constants for the NL2SQL agent."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Server & Model ---
# llama-server endpoint (local or WSL gateway to Windows host)
# Run: llama-server -hf Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF:Q4_K_M --port 8080 -c 8192
LLAMA_SERVER = "http://localhost:8080/v1"

# Model alias as passed to llama-server --alias
# Qwen 3.5 9B distilled from DeepSeek V4 Flash (Q4_K_M, 5.4GB, Apache 2.0)
MODEL_NAME = "qwen3.5-9b-dsv4"

DB_PATH = str(PROJECT_ROOT / "finance.duckdb")

MAX_TRIPS = 5
N_CTX = 8192
CTX_THRESHOLD = 0.9

SYSTEM_PROMPT = """\
You are a financial database agent with access to a DuckDB database of banking data.
Use the available functions to answer questions.

Workflow for every question:
1. Call load_skill to get the relevant table schemas and example queries.
2. Call run_sql to execute your query against the database.
3. Call explain to present the results in natural language.

If run_sql returns an error:
- Call schema_check to verify column names exist.
- Fix the query and call run_sql again.

If a query returns 0 rows, explain that no data matched the criteria.
Always call load_skill first so you know the correct table and column names.
"""
