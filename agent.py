"""Agent loop — orchestrate the NL2SQL pipeline with tool calling and self-correction."""

import json
import re
from typing import Any

import duckdb
from openai import OpenAI

from config import DB_PATH, LLAMA_SERVER, MAX_TRIPS, MODEL_NAME, SYSTEM_PROMPT
from skills import SKILLS, schema_lookup
from tools import TOOLS


def _get_client() -> OpenAI:
    return OpenAI(base_url=LLAMA_SERVER, api_key="not-needed")


def _parse_tool_call(response: Any) -> dict | None:
    """Extract the first tool call from an OpenAI chat completion response."""
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
    """Execute SQL against DuckDB. Returns {columns, rows, error, row_count}."""
    conn = duckdb.connect(DB_PATH)
    try:
        # Strip any markdown backticks that the model might have emitted
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
    """Format query results as a readable string for feeding back to the model."""
    if not rows:
        return "Query returned 0 rows."
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows[:50]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > 50:
        lines.append(f"... ({len(rows) - 50} more rows)")
    return "\n".join(lines)


def _classify_question(
    question: str, client: OpenAI
) -> tuple[str | None, dict | None]:
    """
    Ask the model to classify the question and load the appropriate skill.
    Returns (skill_name, tool_call) or (None, error_info).
    """
    classify_prompt = f"""\
Classify this question and load the most relevant skill.

Question: {question}

Available skills:
- loan-analysis: questions about loans, interest rates, remaining balances
- customer-insights: questions about customers, segments, demographics
- transaction-analysis: questions about spending, categories, dates
- account-overview: questions about account balances, types, currencies

Call load_skill with the ONE most relevant skill. If the question spans
multiple domains, pick the primary one."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": classify_prompt},
        ],
        tools=TOOLS,
        temperature=0.0,
        max_tokens=200,
    )
    tc = _parse_tool_call(response)
    if tc and tc["name"] == "load_skill":
        return tc["args"].get("skill"), tc
    return None, tc


def run_agent(question: str, on_step=None) -> dict:
    """
    Run the full agent loop for a question.

    Args:
        question: Natural language question about the financial database.
        on_step: Optional callback(step_name, data) for UI progress updates.

    Returns:
        dict with keys: answer, sql, results, retries, trips, steps
    """
    client = _get_client()
    steps = []
    retries = 0
    trips_used = 0

    def _step(name: str, data: Any = None):
        steps.append({"name": name, "data": data})
        if on_step:
            on_step(name, data)

    # Step 1: Classify and load skill
    _step("classify", question)
    skill_name, tc = _classify_question(question, client)
    trips_used += 1
    _step("skill_loaded", skill_name)

    if skill_name is None:
        # Model didn't call load_skill — push forward with a safe default
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

    # Step 2+: Tool loop
    while trips_used < MAX_TRIPS:
        trips_used += 1
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
            max_tokens=800,
        )
        tc = _parse_tool_call(response)

        if tc is None:
            # Model responded with text instead of a tool call — extract it
            text = response.choices[0].message.content or ""
            final_answer = text
            _step("text_response", text)
            break

        _step(f"tool:{tc['name']}", tc["args"])

        if tc["name"] == "load_skill":
            skill = tc["args"].get("skill", "")
            if skill in SKILLS:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": "load_skill", "arguments": json.dumps(tc["args"])},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": SKILLS[skill],
                })
                _step("skill_injected", skill)
            continue

        elif tc["name"] == "run_sql":
            sql = tc["args"].get("sql", "")
            sql_history.append(sql)
            result = _execute_sql(sql)

            if result["error"]:
                # Auto-inject duckdb-rules on error if not already present
                if "duckdb-rules" not in skill_name:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": "run_sql", "arguments": json.dumps(tc["args"])},
                        }],
                    })
                    error_content = f"Error: {result['error']}\n\n{SKILLS['duckdb-rules']}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": error_content,
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": "run_sql", "arguments": json.dumps(tc["args"])},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Error: {result['error']}",
                    })
                retries += 1
                _step("sql_error", result["error"])
            else:
                formatted = _format_rows(result["columns"], result["rows"])
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": "run_sql", "arguments": json.dumps(tc["args"])},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": formatted,
                })
                _step("sql_success", {"columns": result["columns"], "row_count": result["row_count"]})
            continue

        elif tc["name"] == "schema_check":
            table = tc["args"].get("table", "")
            columns = schema_lookup(table)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": "schema_check", "arguments": json.dumps(tc["args"])},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": columns,
            })
            _step("schema_checked", table)
            continue

        elif tc["name"] == "explain":
            final_answer = tc["args"].get("text", "")
            _step("explain", final_answer)
            break

    # Build return value
    last_sql = sql_history[-1] if sql_history else None
    last_result = None
    if last_sql:
        last_result = _execute_sql(last_sql)

    return {
        "answer": final_answer or "Agent could not generate a response within the step limit.",
        "sql": last_sql,
        "results": last_result,
        "sql_history": sql_history,
        "retries": retries,
        "trips": trips_used,
        "steps": steps,
        "skill_used": skill_name,
    }
