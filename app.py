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


def _render_result(result: dict, key_suffix: str = ""):
    """Render agent result: answer, SQL, table, chart, trace."""
    answer = result.get("answer", "")
    if answer:
        st.markdown(answer)

    sql = result.get("sql")
    if sql:
        with st.expander("SQL", expanded=False):
            st.code(sql, language="sql")

    res = result.get("results")
    if res and res.get("rows"):
        df = pd.DataFrame(res["rows"], columns=res["columns"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if numeric_cols and len(df) <= 50:
            with st.expander("Chart", expanded=False):
                non_num = [c for c in df.columns if c not in numeric_cols]
                x_default = non_num[0] if non_num else df.columns[0]
                y_default = numeric_cols[0]
                x_col = st.selectbox(
                    "X-axis", df.columns.tolist(),
                    index=df.columns.tolist().index(x_default),
                    key=f"x_{key_suffix}",
                )
                y_col = st.selectbox(
                    "Y-axis", numeric_cols,
                    index=numeric_cols.index(y_default),
                    key=f"y_{key_suffix}",
                )
                try:
                    st.bar_chart(df.set_index(x_col)[y_col])
                except Exception:
                    st.caption("Chart not applicable for this result shape")

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "results.csv", "text/csv", key=f"dl_{key_suffix}")

    elif res and res.get("error"):
        st.error(f"SQL Error: {res['error']}")
    elif not answer:
        st.info("No results found.")

    with st.expander(f"Trace ({result.get('trips', '?')} trips, {result.get('retries', 0)} retries)", expanded=False):
        for s in result.get("steps", []):
            st.text(f"[{s['name']}] {s.get('data', '')}")


def _handle_query(question: str) -> dict:
    """Run the agent and render the result. Returns the result dict."""
    with st.spinner(""):
        result = run_agent(question)
    _render_result(result, key_suffix=str(hash(question)))
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
            _render_result(msg["data"], key_suffix=f"hist_{hash(msg['content'])}")
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
