"""Agent loop using text-based tool-calling (<tool_call>/<tool_response> format).

Model-agnostic: works with any model that can follow format instructions.
Tested with Hermes-2-Pro 8B and Qwen 3.5 9B.
"""

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

# Match balanced { ... } blocks for bare JSON tool calls
BRACE_BLOCK_RE = re.compile(
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL
)

# Match <!-- THINKING -->...<!-- ANSWER --> reasoning wrapper (evaluation prompt format)
THINKING_BLOCK_RE = re.compile(
    r"<!--\s*THINKING\s*-->.*?<!--\s*ANSWER\s*-->", re.DOTALL
)

# Match <think>...</think> reasoning tags (Qwen native chat template format)
QWEN_THINK_RE = re.compile(
    r"<think>.*?</think>", re.DOTALL
)

# Match JSON array [...] for tool-call array format
JSON_ARRAY_RE = re.compile(
    r"\[\s*\{.*?\}\s*\]", re.DOTALL
)


def _strip_thinking(text: str) -> str:
    """Strip reasoning/thinking blocks from model output.

    Handles two formats:
    - <!-- THINKING -->...<!-- ANSWER -->  (evaluation prompt)
    - <think>...</think>                   (Qwen native chat template)
    """
    text = THINKING_BLOCK_RE.sub("", text)
    text = QWEN_THINK_RE.sub("", text)
    return text.strip()


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
    """Parse tool call from model output. Format-agnostic — handles:

    1. <tool_call>{"name":..., "arguments":...}</tool_call>  (Hermes native)
    2. {"name":..., "arguments":...}                          (bare JSON object)
    3. [{"tool":..., "args":...}]                            (Qwen/DeepSeek array)
    4. <!-- THINKING -->...<!-- ANSWER -->[...]               (reasoning wrapper)

    All normalized to {"name": str, "args": dict}.
    """
    # Strip reasoning wrapper if present (Qwen3.5-DeepSeek-V4-Flash)
    text = _strip_thinking(text)

    # --- Strategy 1: <tool_call> wrapper (Hermes native) ---
    match = TOOL_CALL_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            tool_name = data.get("tool") or data.get("name")
            tool_args = data.get("args") or data.get("arguments", {})
            if tool_name:
                return {"name": tool_name, "args": tool_args}
        except json.JSONDecodeError:
            pass

    # --- Strategy 2: JSON array [{...}] (Qwen/DeepSeek eval format) ---
    arr_match = JSON_ARRAY_RE.search(text)
    if arr_match:
        try:
            arr = json.loads(arr_match.group(0))
            if isinstance(arr, list) and len(arr) > 0:
                item = arr[0]
                tool_name = item.get("tool") or item.get("name")
                tool_args = item.get("args") or item.get("arguments", {})
                if tool_name:
                    return {"name": tool_name, "args": tool_args}
        except json.JSONDecodeError:
            pass

    # --- Strategy 3: bare JSON object {"name":..., "arguments":...} ---
    for block in BRACE_BLOCK_RE.findall(text):
        try:
            data = json.loads(block.strip())
            tool_name = data.get("tool") or data.get("name")
            tool_args = data.get("args") or data.get("arguments", {})
            if tool_name:
                return {"name": tool_name, "args": tool_args}
        except json.JSONDecodeError:
            continue

    return None


def _execute_sql(sql: str) -> dict:
    conn = duckdb.connect(DB_PATH)
    try:
        sql = re.sub(r"^```sql\s*", "", sql.strip())
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        # Deduplicate columns from JOINs (e.g., SELECT * produces duplicate customer_id)
        seen = {}
        deduped = []
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
        max_tokens=300,   # Increased — reasoning model may output thinking first
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


def run_agent(question: str, on_step=None, context_history: list[dict] | None = None) -> dict:
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
    ]

    # Inject previous Q&A pairs as multi-turn context (questions + sql + explain outputs)
    # Injected as "user" role so the system prompt's tool-calling mandate stays authoritative
    if context_history:
        history_block = "Previous conversation:\n"
        for i, entry in enumerate(context_history, 1):
            history_block += f"\n[Q{i}] {entry['question']}\n"
            history_block += f"[SQL{i}] {entry['sql']}\n"
            history_block += f"[A{i}] {entry['answer']}\n"
        history_block += "\n(Use the above Q&A context to resolve follow-up references like 'them', 'it', 'those'. Reuse SQL WHERE clauses from previous queries when composing new ones.)"
        messages.append({"role": "user", "content": history_block})

    messages.append({"role": "user", "content": question})

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
            max_tokens=1500,   # Increased from 800 — model may output reasoning first
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
