"""Agent loop using native OpenAI function-calling (tools API).

Designed for Qwen3.5-9B-DeepSeek-V4-Flash via llama.cpp server.
The server's chat template handles all tool-call formatting natively.
No regex parsing, no format guessing, no thinking-stripping needed.
"""

import json
from typing import Any

import duckdb
from openai import OpenAI

from config import (
    CTX_THRESHOLD,
    DB_PATH,
    LLAMA_SERVER,
    MAX_TRIPS,
    MODEL_NAME,
    N_CTX,
    SYSTEM_PROMPT,
)
from skills import SKILLS, schema_lookup


# --- Tool Definitions (OpenAI function-calling format) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load a domain skill card with table schemas and example queries. Always call this first so you know the available tables and columns before writing SQL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": [
                            "loan-analysis",
                            "customer-insights",
                            "transaction-analysis",
                            "account-overview",
                        ],
                        "description": "Domain skill to load",
                    }
                },
                "required": ["skill"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a DuckDB SQL query against the financial database. Returns formatted rows or an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "DuckDB SQL query to execute",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schema_check",
            "description": "Look up the columns and types of a specific table. Use when run_sql returns an error to verify column names exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name to inspect (customers, accounts, transactions, or loans)",
                    }
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain",
            "description": "Present the final answer to the user in natural language. Call this as the LAST step — the conversation ends after this call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Natural language summary of the query results",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


# --- Helpers ---

def _get_client() -> OpenAI:
    return OpenAI(base_url=LLAMA_SERVER, api_key="not-needed")


def _execute_sql(sql: str) -> dict:
    """Execute SQL against DuckDB. Returns {columns, rows, error, row_count}."""
    conn = duckdb.connect(DB_PATH)
    try:
        sql_clean = sql.strip()
        for prefix in ("```sql\n", "```\n", "```sql", "```"):
            if sql_clean.startswith(prefix):
                sql_clean = sql_clean[len(prefix) :]
        if sql_clean.endswith("```"):
            sql_clean = sql_clean[:-3]
        result = conn.execute(sql_clean.strip())
        columns = [desc[0] for desc in result.description]
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for c in columns:
            if c in seen:
                seen[c] += 1
                deduped.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                deduped.append(c)
        rows = result.fetchall()
        conn.close()
        return {"columns": deduped, "rows": rows, "error": None, "row_count": len(rows)}
    except Exception as e:
        conn.close()
        return {"columns": [], "rows": [], "error": str(e), "row_count": 0}


def _format_rows(columns: list[str], rows: list) -> str:
    """Format query results as a pipe-delimited text table."""
    if not rows:
        return "Query returned 0 rows."
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows[:50]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > 50:
        lines.append(f"... ({len(rows) - 50} more rows)")
    return "\n".join(lines)


def _execute_tool(name: str, args: dict) -> str:
    """Execute a single tool call and return the result string."""
    if name == "load_skill":
        skill = args.get("skill", "")
        if skill in SKILLS:
            return SKILLS[skill] + "\n\nNow call run_sql with your SQL query."
        return f"Unknown skill: {skill}. Available: {list(SKILLS.keys())}"

    elif name == "run_sql":
        sql = args.get("sql", "")
        result = _execute_sql(sql)
        if result["error"]:
            return (
                f"Error: {result['error']}\n\n"
                f"{SKILLS['duckdb-rules']}\n\n"
                f"Fix the query and call run_sql again, or call schema_check to verify columns."
            )
        formatted = _format_rows(result["columns"], result["rows"])
        return formatted + "\n\nNow call explain to summarize these results."

    elif name == "schema_check":
        table = args.get("table", "")
        columns = schema_lookup(table)
        return columns + "\n\nNow fix your query and call run_sql."

    elif name == "explain":
        return args.get("text", "")

    return f"Unknown tool: {name}"


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


def _maybe_compress(messages: list[dict], client: OpenAI) -> tuple[list[dict], bool]:
    if len(messages) <= 8:
        return messages, False
    if _estimate_tokens(messages) > int(N_CTX * CTX_THRESHOLD):
        return [messages[0]] + messages[-7:], True
    return messages, False


# --- Main Agent Loop ---

def run_agent(
    question: str,
    on_step: Any = None,
    context_history: list[dict] | None = None,
) -> dict:
    """Run the NL2SQL agent with native function calling.

    Returns:
        dict with keys: answer, sql, results, sql_history, retries, trips, steps, skill_used
    """
    client = _get_client()
    steps: list[dict] = []
    retries = 0
    trips_used = 0
    sql_history: list[str] = []
    final_answer: str | None = None
    skill_used = "customer-insights"

    def _step(name: str, data: Any = None):
        steps.append({"name": name, "data": data})
        if on_step:
            on_step(name, data)

    _step("start", question)

    # Build initial messages
    full_system = SYSTEM_PROMPT + "\n\n" + SKILLS["duckdb-rules"]
    messages: list[dict] = [{"role": "system", "content": full_system}]

    # Inject previous Q&A pairs for multi-turn follow-up resolution
    if context_history:
        history_block = "Previous conversation:\n"
        for i, entry in enumerate(context_history, 1):
            history_block += f"\n[Q{i}] {entry['question']}\n"
            history_block += f"[SQL{i}] {entry.get('sql', '')}\n"
            history_block += f"[A{i}] {entry.get('answer', '')}\n"
        history_block += (
            "\n(Use the above to resolve references like 'them', 'it', 'those'."
            " Reuse SQL WHERE clauses from previous queries when composing new ones.)"
        )
        messages.append({"role": "user", "content": history_block})

    messages.append({"role": "user", "content": question})

    while trips_used < MAX_TRIPS:
        trips_used += 1
        messages, compressed = _maybe_compress(messages, client)
        if compressed:
            _step("context_compressed", True)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1500,
        )
        msg = response.choices[0].message

        # No tool calls — model responded with text (final answer or error)
        if not msg.tool_calls:
            if msg.content and msg.content.strip():
                final_answer = msg.content.strip()
                _step("text_response", final_answer)
            else:
                final_answer = "Agent produced no output."
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        # Append the assistant message with its tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and feed results back
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            _step(f"tool:{tool_name}", tool_args)

            if tool_name == "load_skill":
                skill_used = tool_args.get("skill", skill_used)

            result_text = _execute_tool(tool_name, tool_args)

            if tool_name == "run_sql":
                sql_history.append(tool_args.get("sql", ""))

            if tool_name == "explain":
                final_answer = tool_args.get("text", "")

            if tool_name == "run_sql" and "Error:" in result_text:
                retries += 1
                _step("sql_error", result_text[:200])
            elif tool_name == "run_sql":
                _step("sql_success", {"preview": result_text[:200]})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

        # If explain was called, we're done
        if final_answer is not None:
            break

    last_sql = sql_history[-1] if sql_history else None
    last_result = _execute_sql(last_sql) if last_sql else None

    return {
        "answer": final_answer or "Agent could not respond within the step limit.",
        "sql": last_sql,
        "results": last_result,
        "sql_history": sql_history,
        "retries": retries,
        "trips": trips_used,
        "steps": steps,
        "skill_used": skill_used,
    }
