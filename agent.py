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

def _summarize_results(results: dict | None) -> str:
    """Produce a compact summary of query results for multi-turn context.

    Returns a string like '16 rows, e.g.: KL (12), Penang (8)' or '3 rows'.
    """
    if not results or not results.get("rows"):
        return "0 rows"
    rows = results["rows"]
    cols = results.get("columns", [])
    count = results.get("row_count", len(rows))
    summary = f"{count} row{'s' if count != 1 else ''}"
    # Show a few example values for context (up to 3 rows, first 3 columns)
    if count > 0 and cols:
        examples = []
        for row in rows[:3]:
            vals = [str(v) for v in row[:3]]
            examples.append("(" + ", ".join(vals) + ")")
        summary += ", e.g.: " + " ".join(examples)
    return summary


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
        history_block = "Previous conversation (for context — resolve references like 'them', 'it', 'those'):\n"
        for i, entry in enumerate(context_history, 1):
            history_block += f"\n[Q{i}] {entry['question']}\n"
            skill = entry.get("skill", "")
            if skill:
                history_block += f"[Skill] {skill}\n"
            results_summary = entry.get("results_summary", "")
            if results_summary:
                history_block += f"[Results] {results_summary}\n"
            history_block += f"[A{i}] {entry['answer']}\n"
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


# --- Streaming Agent Loop ---

def _stream_llm_response(client: OpenAI, messages: list[dict], tools: list[dict],
                         temperature: float = 0.0, max_tokens: int = 1500):
    """Stream an LLM response and yield events as they arrive.

    Yields:
        {"type": "text_chunk", "content": "..."}     — partial text token
        {"type": "tool_call", "id": ..., "name": ..., "args": {...}}  — complete tool call
        {"type": "stream_end", "content": "...", "tool_calls": [...]}
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    content = ""
    tool_calls_acc: dict[int, dict] = {}  # index → {id, function: {name, arguments}}

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        # Streaming text (may be reasoning, thinking, or final answer)
        if delta.content:
            content += delta.content
            yield {"type": "text_chunk", "content": delta.content}

        # Accumulate tool calls — arguments arrive in pieces across chunks
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_acc[idx]
                if tc.id:
                    entry["id"] = tc.id
                if tc.function and tc.function.name:
                    entry["function"]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    entry["function"]["arguments"] += tc.function.arguments

        finish = chunk.choices[0].finish_reason if chunk.choices else None
        if finish:
            break

    # Yield each complete tool call (arguments now fully assembled)
    assembled: list[dict] = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        assembled.append(tc)
        yield {
            "type": "tool_call",
            "id": tc["id"],
            "name": tc["function"]["name"],
            "args": args,
        }

    yield {"type": "stream_end", "content": content, "tool_calls": assembled}


def run_agent_stream(
    question: str,
    context_history: list[dict] | None = None,
):
    """Run the NL2SQL agent with streaming — yields events for live UI updates.

    Yields events:
        {"type": "step", "name": "start", "data": question}
        {"type": "tool_call", "name": ..., "id": ..., "args": {...}}
        {"type": "tool_result", "name": ..., "content": ..., "results": {...} | None}
        {"type": "sql_error", "error": "..."}
        {"type": "text_chunk", "content": "..."}
        {"type": "done", "data": result_dict}
    """
    client = _get_client()
    retries = 0
    trips_used = 0
    sql_history: list[str] = []
    final_answer: str | None = None
    skill_used = "customer-insights"

    yield {"type": "step", "name": "start", "data": question}

    # Build initial messages
    full_system = SYSTEM_PROMPT + "\n\n" + SKILLS["duckdb-rules"]
    messages: list[dict] = [{"role": "system", "content": full_system}]

    # Inject multi-turn context
    if context_history:
        history_block = "Previous conversation (for context — resolve references like 'them', 'it', 'those'):\n"
        for i, entry in enumerate(context_history, 1):
            history_block += f"\n[Q{i}] {entry['question']}\n"
            skill = entry.get("skill", "")
            if skill:
                history_block += f"[Skill] {skill}\n"
            results_summary = entry.get("results_summary", "")
            if results_summary:
                history_block += f"[Results] {results_summary}\n"
            history_block += f"[A{i}] {entry['answer']}\n"
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
            yield {"type": "step", "name": "context_compressed"}

        # Stream the LLM response
        tool_calls_this_turn: list[dict] = []
        for event in _stream_llm_response(client, messages, TOOLS):
            if event["type"] == "text_chunk":
                yield event  # pass through streaming text

            elif event["type"] == "tool_call":
                tool_calls_this_turn.append(event)
                tc_name = event["name"]
                tc_args = event["args"]

                # Yield tool_call for UI
                yield event

                # Track state
                if tc_name == "load_skill":
                    skill_used = tc_args.get("skill", skill_used)

                # Execute the tool
                result_text = _execute_tool(tc_name, tc_args)

                # Stash result so stream_end can reuse it (avoid double execution)
                event["_result"] = result_text

                if tc_name == "run_sql":
                    sql_history.append(tc_args.get("sql", ""))

                if tc_name == "explain":
                    final_answer = tc_args.get("text", "")

                # Determine if SQL error
                if tc_name == "run_sql" and "Error:" in result_text:
                    retries += 1
                    yield {"type": "sql_error", "error": result_text[:300]}

                # Yield tool result for UI (include parsed results for SQL queries)
                tool_result_event: dict = {
                    "type": "tool_result",
                    "name": tc_name,
                    "id": event.get("id", ""),
                    "content": result_text,
                    "results": None,
                }
                if tc_name == "run_sql":
                    sql_args = tc_args.get("sql", "")
                    exec_result = _execute_sql(sql_args)
                    tool_result_event["results"] = exec_result if not exec_result.get("error") else None

                yield tool_result_event

            elif event["type"] == "stream_end":
                # Build assistant message from this turn
                assistant_content = event.get("content", "")
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": assistant_content,
                }
                if event.get("tool_calls"):
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in event["tool_calls"]
                    ]
                messages.append(assistant_msg)

                # Append tool responses (reuse stashed results — no double execution)
                for tc in tool_calls_this_turn:
                    tc_name = tc["name"]
                    tc_id = tc.get("id", "")
                    result_text = tc.get("_result",
                        _execute_tool(tc_name, tc.get("args", {})))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_text,
                    })

                # If no tool calls this turn, model gave a text-only response
                if not tool_calls_this_turn:
                    if assistant_content and assistant_content.strip():
                        final_answer = assistant_content.strip()
                    else:
                        final_answer = "Agent produced no output."
                    break

        # Check if we're done
        if final_answer is not None:
            break

    last_sql = sql_history[-1] if sql_history else None
    last_result = _execute_sql(last_sql) if last_sql else None

    result = {
        "answer": final_answer or "Agent could not respond within the step limit.",
        "sql": last_sql,
        "results": last_result,
        "sql_history": sql_history,
        "retries": retries,
        "trips": trips_used,
        "skill_used": skill_used,
    }
    yield {"type": "done", "data": result}
