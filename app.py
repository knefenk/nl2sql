"""Streamlit chat UI for the NL2SQL agent."""

import streamlit as st
import pandas as pd

from agent import run_agent

st.set_page_config(page_title="NL2SQL Agent", layout="wide")

SAMPLE_QUESTIONS = [
    "How many customers in each segment?",
    "Top 5 customers by total account balance",
    "Total transaction amount per category",
    "Customers with loans over RM 200,000",
    "Which city has the most customers?",
    "Show all transactions for Alice Tan",
    "Average loan interest rate by loan type",
    "Accounts with balance above RM 100,000",
    "Total outstanding loan balance per customer segment",
    "How many customers joined in 2024?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


def _render_result(result: dict):
    """Render agent tool calls chronologically, explain at bottom."""
    steps = result.get("steps", [])
    table_shown = False

    for step in steps:
        name = step["name"]
        data = step.get("data")

        if name == "skill_loaded":
            skill = data or "unknown"
            with st.expander(f"load_skill: {skill}", expanded=False):
                from skills import SKILLS
                content = SKILLS.get(skill, "Skill not found.")
                st.text(content[:1500])
                if len(content) > 1500:
                    st.caption("(truncated)")

        elif name == "fallback_skill":
            st.caption(f"Classification failed, using fallback: {data}")

        elif name == "tool:run_sql":
            sql = data.get("sql", "") if isinstance(data, dict) else str(data)
            st.code(sql, language="sql")

        elif name == "sql_error":
            st.error(f"SQL error: {data}")

        elif name == "sql_success" and not table_shown:
            table_shown = True
            res = result.get("results")
            if res and res.get("rows"):
                df = pd.DataFrame(res["rows"], columns=res["columns"])
                st.caption(f"{len(df)} rows")
                st.dataframe(df, use_container_width=True, hide_index=True)

        elif name == "tool:schema_check":
            table = data.get("table", "") if isinstance(data, dict) else str(data)
            st.caption(f"Checking schema for: {table}")

        elif name == "context_compressed":
            st.caption("Context compressed to stay within token limit.")

    # Explanation at the bottom
    answer = result.get("answer", "")
    if answer and answer != "Agent could not respond within the step limit.":
        st.markdown(answer)
    elif not (result.get("results") and result["results"].get("rows")):
        if result.get("retries", 0) > 0:
            st.error(f"Query failed after {result['retries']} retries.")
        else:
            st.info("No results found.")

    # Footer
    st.caption(
        f"skill: {result.get('skill_used', '?')}  |  "
        f"trips: {result.get('trips', 0)}  |  "
        f"retries: {result.get('retries', 0)}"
    )


def _handle_query(question: str) -> dict:
    with st.spinner(""):
        result = run_agent(question, context_history=st.session_state.qa_history[-5:])
    _render_result(result)

    # Record Q&A for multi-turn context (questions + sql + explain outputs)
    # Skip entries with SQL errors — those answers are likely wrong
    answer = result.get("answer", "")
    sql = result.get("sql", "")
    retries = result.get("retries", 0)
    if answer and answer != "Agent could not respond within the step limit." and retries == 0 and sql:
        st.session_state.qa_history.append({
            "question": question,
            "answer": answer,
            "sql": sql,
        })

    return result


# --- Sidebar ---
with st.sidebar:
    st.header("Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q[:30]}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    st.header("Context")
    if st.button("Clear conversation context", use_container_width=True):
        st.session_state.qa_history = []
        st.rerun()
    st.caption(f"{len(st.session_state.qa_history)} previous Q&A pairs stored")

    st.divider()
    st.header("About")
    st.markdown(
        "NL2SQL agent powered by Hermes-2-Pro 8B running locally via llama.cpp. "
        "Uses domain skill cards, self-correcting SQL generation, and schema-aware retry."
    )

# --- Header ---
st.title("NL2SQL Agent")
st.caption("Ask questions about the financial database in plain English.")

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_result(msg["data"])
        else:
            st.markdown(msg["content"])

# --- Chat input ---
if question := st.chat_input("Ask a question about the financial database..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result = _handle_query(question)
    st.session_state.messages.append({"role": "assistant", "content": question, "data": result})
    st.rerun()

# --- Sample question click ---
if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        result = _handle_query(q)
    st.session_state.messages.append({"role": "assistant", "content": q, "data": result})
    st.rerun()
