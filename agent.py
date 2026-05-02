"""Agent loop with tool calling, self-correction, and context management."""

import json
import re
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
from tools import TOOLS


def _get_client() -> OpenAI:
    return OpenAI(base_url=LLAMA_SERVER, api_key="not-needed")


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: 1 token per 4 characters."""
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


def _summarize_context(messages: list[dict], client: OpenAI) -> list[dict]:
    """
    Compress the middle of the conversation when near context limit.
    Keeps the system message and last 3 turns, summarizes the rest.
    """
    if len(messages) <= 6:
        return messages

    middle = messages[1:-4]
    if not middle:
        return messages

    summary_input = "Summarize this agent conversation trace. Keep all SQL queries, errors, results, and corrections.\n\n"
    for m in middle:
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                content = f"[tool_call: {tc['function']['name']}]"
        summary_input += f"[{role}] {str(content)[:300]}\n"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You compress conversation traces. Be concise."},
            {"role": "user", "content": summary_input},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    summary = response.choices[0].message.content or ""

    return [messages[0]] + [
        {"role": "system", "content": f"Earlier trace (compressed):\n{summary}"}
    ] + messages[-4:]


def _parse_tool_call(response: Any) -> dict | None:
    if not response.choices:
        return None
    choice = response.choices[0]
    if not choice.message.tool_calls:
        return None
    tc = choice.message.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        return None
    return {"id": tc.id, "name": tc.function.name, "args": args}


def _execute_sql(sql: str) -> dict:
    conn = duckdb.connect(DB_PATH)
    try:
        sql = re.sub(r"^```sql\s*", "", sql.strip())
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        conn.close()
        return {"columns": columns, "rows": rows, "error": None, "row_count": len(rows)}
    except Exception as e:
        conn.close()
        return {"columns": [], "rows": [], "error": str(e), "row_count": 0}


def _format_rows(columns: list[str], rows: list) -> str:
    if not rows:
        return "Query returned 0 rows."
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows[:50]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > 50:
        lines.append(f"... ({len(rows) - 50} more rows)")
    return "\n".join(lines)


def _add_tool_message(messages: list[dict], tc_id: str, name: str, args: dict, content: str) -> None:
    """Append assistant tool_call + tool result to the message list."""
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": content,
    })


def _maybe_compress(messages: list[dict], client: OpenAI) -> tuple[list[dict], bool]:
    """Compress context if near limit. Returns (messages, was_compressed)."""
    if _estimate_tokens(messages) > int(N_CTX * CTX_THRESHOLD):
        return _summarize_context(messages, client), True
    return messages, False


def _classify_question(question: str, client: OpenAI) -> tuple[str | None, dict | None]:
    classify_prompt = f"""\
Classify this question and load the most relevant skill.

Question: {question}

Available skills:
- loan-analysis: questions about loans, interest rates, remaining balances
- customer-insights: questions about customers, segments, demographics
- transaction-analysis: questions about spending, categories, dates
- account-overview: questions about account balances, types, currencies

Call load_skill with the ONE most relevant skill."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": classify_prompt}],
        tools=TOOLS,
        temperature=0.0,
        max_tokens=200,
    )
    tc = _parse_tool_call(response)
    if tc and tc["name"] == "load_skill":
        return tc["args"].get("skill"), tc
    return None, tc


def run_agent(question: str, on_step=None) -> dict:
    client = _get_client()
    steps = []
    retries = 0
    trips_used = 0

    def _step(name: str, data: Any = None):
        steps.append({"name": name, "data": data})
        if on_step:
            on_step(name, data)

    _step("classify", question)
    skill_name, tc = _classify_question(question, client)
    trips_used += 1
    _step("skill_loaded", skill_name)

    if skill_name is None:
        skill_name = "customer-insights"
        _step("fallback_skill", skill_name)

    skill_text = SKILLS.get(skill_name, SKILLS["customer-insights"])
    system_prompt = SYSTEM_PROMPT + "\n\n" + skill_text + "\n\n" + SKILLS["duckdb-rules"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    sql_history = []
    final_answer = None

    while trips_used < MAX_TRIPS:
        trips_used += 1
        messages, compressed = _maybe_compress(messages, client)
        if compressed:
            _step("context_compressed", True)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
            max_tokens=800,
        )
        tc = _parse_tool_call(response)

        if tc is None:
            text = response.choices[0].message.content or ""
            final_answer = text
            _step("text_response", text)
            break

        _step(f"tool:{tc['name']}", tc["args"])

        if tc["name"] == "load_skill":
            skill = tc["args"].get("skill", "")
            if skill in SKILLS:
                _add_tool_message(messages, tc["id"], "load_skill", tc["args"], SKILLS[skill])
                _step("skill_injected", skill)
            continue

        elif tc["name"] == "run_sql":
            sql = tc["args"].get("sql", "")
            sql_history.append(sql)
            result = _execute_sql(sql)

            if result["error"]:
                rules_text = SKILLS["duckdb-rules"] if "duckdb-rules" not in str(messages[-6:]) else ""
                error_content = f"Error: {result['error']}"
                if rules_text:
                    error_content += f"\n\n{rules_text}"
                _add_tool_message(messages, tc["id"], "run_sql", tc["args"], error_content)
                retries += 1
                _step("sql_error", result["error"])
            else:
                formatted = _format_rows(result["columns"], result["rows"])
                _add_tool_message(messages, tc["id"], "run_sql", tc["args"], formatted)
                _step("sql_success", {"columns": result["columns"], "row_count": result["row_count"]})
            continue

        elif tc["name"] == "schema_check":
            table = tc["args"].get("table", "")
            columns = schema_lookup(table)
            _add_tool_message(messages, tc["id"], "schema_check", tc["args"], columns)
            _step("schema_checked", table)
            continue

        elif tc["name"] == "explain":
            final_answer = tc["args"].get("text", "")
            _step("explain", final_answer)
            break

    last_sql = sql_history[-1] if sql_history else None
    last_result = None
    if last_sql:
        last_result = _execute_sql(last_sql)

    return {
        "answer": final_answer or "Agent could not respond within the step limit.",
        "sql": last_sql,
        "results": last_result,
        "sql_history": sql_history,
        "retries": retries,
        "trips": trips_used,
        "steps": steps,
        "skill_used": skill_name,
    }
