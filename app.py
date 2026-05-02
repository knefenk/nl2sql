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


def _render_result(result: dict):
    """Render agent result in a human-readable chat bubble."""
    # Skill loaded
    skill = result.get("skill_used", "unknown")
    trips = result.get("trips", 0)
    retries = result.get("retries", 0)
    st.caption(f"skill: {skill}  |  trips: {trips}  |  retries: {retries}")

    # SQL
    sql = result.get("sql")
    if sql:
        st.code(sql, language="sql")

    # Results table
    res = result.get("results")
    if res and res.get("rows"):
        df = pd.DataFrame(res["rows"], columns=res["columns"])
        st.caption(f"{len(df)} rows")
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif res and res.get("error"):
        st.error(f"SQL Error: {res['error']}")

    # Explanation — the main human-readable answer
    answer = result.get("answer", "")
    if answer:
        st.markdown(answer)
    elif not (res and res.get("rows")):
        st.info("No results found.")


def _handle_query(question: str) -> dict:
    with st.spinner(""):
        result = run_agent(question)
    _render_result(result)
    return result


# --- Sidebar ---
with st.sidebar:
    st.header("Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q[:30]}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

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
