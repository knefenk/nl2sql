"""Streamlit chat UI for the NL2SQL agent — with live streaming."""

import streamlit as st
import pandas as pd

from agent import run_agent_stream, _summarize_results

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
    """Render agent result for historical messages (no streaming)."""
    answer = result.get("answer", "")
    results = result.get("results")

    # Show results table if available
    if results and results.get("rows"):
        df = pd.DataFrame(results["rows"], columns=results["columns"])
        st.caption(f"{len(df)} rows")
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Show SQL if available
    sql = result.get("sql", "")
    if sql:
        with st.expander("SQL", expanded=False):
            st.code(sql, language="sql")

    # Answer text
    if answer and answer != "Agent could not respond within the step limit.":
        st.markdown(answer)
    elif result.get("retries", 0) > 0:
        st.error(f"Query failed after {result['retries']} retries.")
    elif not (results and results.get("rows")):
        st.info("No results found.")

    # Footer
    st.caption(
        f"skill: {result.get('skill_used', '?')}  |  "
        f"trips: {result.get('trips', 0)}  |  "
        f"retries: {result.get('retries', 0)}"
    )


def _handle_query(question: str) -> dict:
    """Run a query with live streaming UI, record context for follow-ups."""
    with st.chat_message("assistant"):
        # ── Live progress container ──
        status = st.status("Thinking...", expanded=True)

        # Dynamic placeholders (outside status — persist after collapse)
        sql_placeholder = st.empty()
        results_placeholder = st.empty()
        answer_placeholder = st.empty()
        footer_placeholder = st.empty()

        answer_text = ""
        result_data: dict = {}

        for event in run_agent_stream(
            question, context_history=st.session_state.qa_history[-10:]
        ):
            ev_type = event["type"]

            # ── Step events ──
            if ev_type == "step":
                if event["name"] == "start":
                    status.write(f"**Processing:** {event.get('data', '')[:100]}...")
                elif event["name"] == "context_compressed":
                    status.write("📦 Context compressed to stay within token limit")

            # ── Tool calls ──
            elif ev_type == "tool_call":
                name = event["name"]
                if name == "load_skill":
                    status.write(f"🔍 Loading domain skill: **{event['args'].get('skill', '?')}**")
                elif name == "run_sql":
                    status.write("📊 Running SQL query...")
                    sql_placeholder.code(event["args"].get("sql", ""), language="sql")
                elif name == "schema_check":
                    status.write(f"🔍 Checking schema: **{event['args'].get('table', '?')}**")
                elif name == "explain":
                    status.write("💬 Generating explanation...")
                    # Clear any prior SQL display since we're done querying
                    sql_placeholder.empty()

            # ── Tool results ──
            elif ev_type == "tool_result":
                name = event["name"]
                if name == "load_skill":
                    status.write("✓ Skill loaded — generating query...")
                elif name == "run_sql":
                    res = event.get("results")
                    if res and res.get("rows"):
                        df = pd.DataFrame(res["rows"], columns=res["columns"])
                        status.write(f"✓ Query returned **{len(df)} rows**")
                        results_placeholder.dataframe(
                            df, use_container_width=True, hide_index=True
                        )
                    elif res:
                        status.write("✓ Query returned 0 rows")
                    else:
                        status.write("✓ Query executed")
                elif name == "schema_check":
                    status.write("✓ Schema checked — regenerating query...")

            # ── SQL errors ──
            elif ev_type == "sql_error":
                status.write(f"❌ SQL error — auto-correcting...")
                # Clear bad SQL and stale results
                sql_placeholder.empty()
                results_placeholder.empty()

            # ── Streaming text (final explain) ──
            elif ev_type == "text_chunk":
                chunk = event["content"]
                # Strip thinking blocks from display
                if not chunk.strip():
                    continue
                answer_text += chunk
                # Show text with cursor while streaming
                answer_placeholder.markdown(answer_text + " ▌")

            # ── Done ──
            elif ev_type == "done":
                result_data = event["data"]
                # Replace cursor with final answer
                final_answer = result_data.get("answer", answer_text)
                if (
                    final_answer
                    and final_answer != "Agent could not respond within the step limit."
                ):
                    answer_placeholder.markdown(final_answer)
                else:
                    if result_data.get("retries", 0) > 0:
                        answer_placeholder.error(
                            f"Query failed after {result_data['retries']} retries."
                        )
                    else:
                        answer_placeholder.info("No results found.")

                status.update(label="Complete ✓", state="complete", expanded=False)

                footer_placeholder.caption(
                    f"skill: {result_data.get('skill_used', '?')}  |  "
                    f"trips: {result_data.get('trips', 0)}  |  "
                    f"retries: {result_data.get('retries', 0)}"
                )

    # ── Record multi-turn context ──
    answer = result_data.get("answer", "")
    sql = result_data.get("sql", "")
    retries = result_data.get("retries", 0)
    if (
        answer
        and answer != "Agent could not respond within the step limit."
        and retries == 0
        and sql
    ):
        st.session_state.qa_history.append(
            {
                "question": question,
                "answer": answer,
                "skill": result_data.get("skill_used", ""),
                "results_summary": _summarize_results(result_data.get("results")),
            }
        )

    return result_data


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

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
        "NL2SQL agent powered by Qwen3.5-9B (DeepSeek V4 distilled) "
        "running locally via llama.cpp. Uses native function calling, "
        "domain skill cards, self-correcting SQL, schema-aware retry, "
        "and live token-by-token streaming."
    )

# ═══════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════

st.title("NL2SQL Agent")
st.caption("Ask questions about the financial database in plain English.")

# ═══════════════════════════════════════════════════════════════
# Chat history
# ═══════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_result(msg["data"])
        else:
            st.markdown(msg["content"])

# ═══════════════════════════════════════════════════════════════
# Chat input
# ═══════════════════════════════════════════════════════════════

if question := st.chat_input("Ask a question about the financial database..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = _handle_query(question)
    st.session_state.messages.append(
        {"role": "assistant", "content": question, "data": result}
    )
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# Sample question click
# ═══════════════════════════════════════════════════════════════

if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    result = _handle_query(q)
    st.session_state.messages.append(
        {"role": "assistant", "content": q, "data": result}
    )
    st.rerun()
