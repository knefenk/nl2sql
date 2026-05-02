"""Agent loop using Hermes-2-Pro native <tool_call> format."""

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


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# Also accept bare JSON: {"name": "...", "arguments": {...}}
BARE_TOOL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]+\}\s*\}', re.DOTALL
)


def _get_client() -> OpenAI:
    return OpenAI(base_url=LLAMA_SERVER, api_key="not-needed")


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


def _summarize_context(messages: list[dict], client: OpenAI) -> list[dict]:
    if len(messages) <= 6:
        return messages

    middle = messages[1:-4]
    if not middle:
        return messages

    trace = ""
    for m in middle:
        role = m.get("role", "?")
        content = str(m.get("content", ""))[:400]
        trace += f"[{role}] {content}\n"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Compress this agent trace. Keep SQL queries, errors, and results. Be concise."},
            {"role": "user", "content": trace},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    summary = response.choices[0].message.content or ""

    return [messages[0]] + [
        {"role": "system", "content": f"Earlier trace (compressed):\n{summary}"}
    ] + messages[-4:]


def _maybe_compress(messages: list[dict], client: OpenAI) -> tuple[list[dict], bool]:
    if _estimate_tokens(messages) > int(N_CTX * CTX_THRESHOLD):
        return _summarize_context(messages, client), True
    return messages, False


def _parse_tool_call(text: str) -> dict | None:
    """Parse tool call from text. Accepts both <tool_call> wrapper and bare JSON."""
    # Try <tool_call> wrapper first
    match = TOOL_CALL_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    else:
        # Try bare JSON: {"name": "...", "arguments": {...}}
        match = BARE_TOOL_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0).strip())
        except json.JSONDecodeError:
            return None
    name = data.get("name", "")
    args = data.get("arguments", {})
    if not name:
        return None
    return {"name": name, "args": args}


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


def _classify_question(question: str, client: OpenAI) -> tuple[str | None, Any]:
    classify_prompt = f"""\
You are a classifier. Respond ONLY with a tool call.

Question: {question}

Functions:
- load_skill(skill: "loan-analysis" | "customer-insights" | "transaction-analysis" | "account-overview")

Pick the ONE most relevant skill. Respond with the function call."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": classify_prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    text = response.choices[0].message.content or ""
    tc = _parse_tool_call(text)
    if tc and tc["name"] == "load_skill":
        return tc["args"].get("skill"), tc
    # Fallback: try to extract skill name from raw text
    skill_match = re.search(r"loan-analysis|customer-insights|transaction-analysis|account-overview", text)
    if skill_match:
        return skill_match.group(0), None
    return None, None


def _append_assistant(messages: list[dict], content: str) -> None:
    messages.append({"role": "assistant", "content": content})


def _feed_tool_response(messages: list[dict], content: str) -> None:
    """Feed a tool response back to the model in Hermes native format."""
    messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})


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
    skill_name, _ = _classify_question(question, client)
    trips_used += 1
    _step("skill_loaded", skill_name)

    if skill_name is None:
        skill_name = "customer-insights"
        _step("fallback_skill", skill_name)

    skill_text = SKILLS.get(skill_name, SKILLS["customer-insights"])
    full_system = SYSTEM_PROMPT + "\n\n" + skill_text + "\n\n" + SKILLS["duckdb-rules"]

    messages = [
        {"role": "system", "content": full_system},
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
            temperature=0.0,
            max_tokens=800,
        )
        text = response.choices[0].message.content or ""
        _append_assistant(messages, text)

        tc = _parse_tool_call(text)

        if tc is None:
            # No tool call found — check if this is a plain text answer
            if text.strip():
                final_answer = text.strip()
                _step("text_response", final_answer)
            else:
                final_answer = "Agent produced no output."
                _step("empty_response", None)
            break

        _step(f"tool:{tc['name']}", tc["args"])

        if tc["name"] == "load_skill":
            skill = tc["args"].get("skill", "")
            if skill in SKILLS:
                _feed_tool_response(messages, SKILLS[skill] + "\n\nNow call run_sql with your SQL query.")
                _step("skill_injected", skill)
            continue

        elif tc["name"] == "run_sql":
            sql = tc["args"].get("sql", "")
            sql_history.append(sql)
            result = _execute_sql(sql)

            if result["error"]:
                content = f"Error: {result['error']}"
                if "duckdb-rules" not in str(messages[-6:]):
                    content += f"\n\n{SKILLS['duckdb-rules']}"
                content += "\n\nFix the query and call run_sql again, or call schema_check to verify columns."
                _feed_tool_response(messages, content)
                retries += 1
                _step("sql_error", result["error"])
            else:
                formatted = _format_rows(result["columns"], result["rows"])
                _feed_tool_response(messages, formatted + "\n\nNow call explain to summarize these results.")
                _step("sql_success", {"columns": result["columns"], "row_count": result["row_count"]})
            continue

        elif tc["name"] == "schema_check":
            table = tc["args"].get("table", "")
            columns = schema_lookup(table)
            _feed_tool_response(messages, columns + "\n\nNow fix your query and call run_sql.")
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
