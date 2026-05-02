"""Streamlit UI for the NL2SQL agent."""

import streamlit as st
import pandas as pd

from agent import run_agent


st.set_page_config(page_title="NL2SQL Agent", page_icon=":bar_chart:", layout="wide")

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

st.title("NL2SQL Agent")
st.caption("DuckDB + llama-3-sqlcoder-8b agentic SQL system")

# Sidebar
with st.sidebar:
    st.header("Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q[:30]}", use_container_width=True):
            st.session_state.question = q

    st.divider()
    st.header("About")
    st.markdown("""
    This agent converts natural language questions into DuckDB SQL queries
    using the llama-3-sqlcoder-8b model running locally via llama.cpp.

    **Architecture:**
    - Skills-based domain routing
    - Self-correcting SQL generation
    - Schema-aware retry on column errors
    """)

# Main
question = st.text_input(
    "Ask a question about the financial database",
    placeholder="e.g., Top 5 customers by loan balance",
    key="question_input",
    label_visibility="collapsed",
)

if "question" in st.session_state and question == "":
    question = st.session_state.pop("question")

if question.strip():
    with st.spinner("Agent is working..."):
        steps_placeholder = st.empty()
        result = run_agent(question)

    # Show agent workflow
    with st.expander("Agent Workflow", expanded=False):
        for step in result["steps"]:
            st.text(f"[{step['name']}] {step.get('data', '')}")

    # Show SQL
    if result["sql"]:
        with st.expander("Generated SQL", expanded=True):
            st.code(result["sql"], language="sql")

    # Show results or error
    res = result["results"]
    if res and res["error"]:
        st.error(f"SQL Error: {res['error']}")
        if result.get("retries", 0) > 0:
            st.info(f"Agent retried {result['retries']} time(s). Check workflow for details.")
    elif res and res["rows"]:
        df = pd.DataFrame(res["rows"], columns=res["columns"])
        st.subheader(f"Results ({len(df)} rows)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if numeric_cols and len(df) <= 50:
            with st.expander("Chart", expanded=False):
                chart_type = st.radio(
                    "Type", ["Bar", "Line", "Area"],
                    horizontal=True,
                    key=f"chart_{hash(question)}",
                )
                non_num = [c for c in df.columns if c not in numeric_cols]
                x_default = non_num[0] if non_num else df.columns[0]
                y_default = numeric_cols[0]
                x_col = st.selectbox(
                    "X-axis", df.columns.tolist(),
                    index=df.columns.tolist().index(x_default),
                    key=f"x_{hash(question)}",
                )
                y_col = st.selectbox(
                    "Y-axis", numeric_cols,
                    index=numeric_cols.index(y_default),
                    key=f"y_{hash(question)}",
                )
                try:
                    chart_df = df.set_index(x_col)[y_col]
                    if chart_type == "Bar":
                        st.bar_chart(chart_df)
                    elif chart_type == "Line":
                        st.line_chart(chart_df)
                    else:
                        st.area_chart(chart_df)
                except Exception:
                    st.caption("Chart not applicable for this result shape")

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "results.csv", "text/csv")
    elif res:
        st.info("No results found for this query.")
    else:
        if result.get("answer"):
            st.info(result["answer"])
        else:
            st.warning("Agent did not produce a SQL query or response.")

    # Footer stats
    st.caption(
        f"Skill: {result.get('skill_used', 'unknown')} | "
        f"Trips: {result.get('trips', 0)}/{5} | "
        f"Retries: {result.get('retries', 0)}"
    )
