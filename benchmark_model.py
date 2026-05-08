"""Benchmark a model for NL2SQL agent: SQL quality + tool-calling reliability.

Tests native function-calling (tools API) — no text parsing.
The llama.cpp server handles all tool-call formatting via the chat template.

Usage:
    python benchmark_model.py [--model MODEL_NAME] [--server URL]
"""

import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# Re-use tool definitions from agent
from agent import TOOLS
from config import SYSTEM_PROMPT
from skills import SKILLS

# --- Test Queries ---
TEST_QUERIES = [
    {
        "id": "Q1",
        "question": "How many customers are there?",
        "sql_must_contain": ["COUNT", "customers"],
        "difficulty": "easy",
        "skill": "customer-insights",
    },
    {
        "id": "Q2",
        "question": "What is the total balance for each account type?",
        "sql_must_contain": ["SUM", "balance", "GROUP BY", "account_type", "accounts"],
        "difficulty": "easy",
        "skill": "account-overview",
    },
    {
        "id": "Q3",
        "question": "List customers who have loans with remaining balance over 100,000 MYR. Show their name and remaining balance.",
        "sql_must_contain": ["JOIN", "customers", "loans", "remaining_balance", "100000"],
        "difficulty": "medium",
        "skill": "loan-analysis",
    },
    {
        "id": "Q4",
        "question": "What is the average transaction amount per category for Retail segment customers?",
        "sql_must_contain": ["AVG", "GROUP BY", "category", "segment", "Retail", "transactions"],
        "difficulty": "hard",
        "skill": "transaction-analysis",
    },
    {
        "id": "Q5",
        "question": "Find customers whose total spending exceeds 100,000 MYR. Show name and total spent.",
        "sql_must_contain": ["SUM", "GROUP BY", "HAVING", "100000"],
        "difficulty": "hard",
        "skill": "transaction-analysis",
    },
]


def check_sql(sql: str, must_contain: list[str]) -> tuple[bool, list[str]]:
    """Check if SQL contains all required patterns (case-insensitive)."""
    sql_upper = sql.upper()
    missing = [p for p in must_contain if p.upper() not in sql_upper]
    return len(missing) == 0, missing


def extract_sql_from_tool_calls(tool_calls: list) -> str | None:
    """Extract SQL from run_sql tool calls in the response."""
    for tc in tool_calls:
        if tc.function.name == "run_sql":
            try:
                args = json.loads(tc.function.arguments)
                return args.get("sql", "")
            except json.JSONDecodeError:
                pass
    return None


def test_sql_generation(client: OpenAI, model: str, queries: list[dict]) -> list[dict]:
    """Test SQL generation by giving the model full context and asking it to call run_sql."""
    results = []
    for q in queries:
        start = time.time()
        try:
            skill_text = SKILLS.get(q["skill"], "")
            system = (
                SYSTEM_PROMPT
                + "\n\n"
                + skill_text
                + "\n\n"
                + SKILLS["duckdb-rules"]
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": q["question"]},
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1500,
            )
            elapsed = time.time() - start
            msg = resp.choices[0].message
            tokens = resp.usage.completion_tokens if resp.usage else 0

            sql = None
            if msg.tool_calls:
                sql = extract_sql_from_tool_calls(msg.tool_calls)

            if sql:
                passed, missing = check_sql(sql, q["sql_must_contain"])
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "difficulty": q["difficulty"],
                    "passed": passed,
                    "sql": sql,
                    "missing_patterns": missing,
                    "elapsed_s": round(elapsed, 2),
                    "completion_tokens": tokens,
                    "tool_calls": [tc.function.name for tc in msg.tool_calls],
                })
            else:
                # Model didn't call run_sql — check if it called explain directly
                tc_names = [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else []
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "difficulty": q["difficulty"],
                    "passed": False,
                    "sql": None,
                    "missing_patterns": q["sql_must_contain"],
                    "elapsed_s": round(elapsed, 2),
                    "completion_tokens": tokens,
                    "tool_calls": tc_names,
                    "raw_content": (msg.content or "")[:200],
                })

        except Exception as e:
            results.append({
                "id": q["id"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "passed": False,
                "error": str(e),
                "elapsed_s": round(time.time() - start, 2),
            })

    return results


def test_tool_calling(client: OpenAI, model: str) -> list[dict]:
    """Test whether the model uses tools appropriately."""
    tests = [
        {
            "id": "TC1",
            "question": "How many customers are in Kuala Lumpur?",
            "description": "Should call load_skill then run_sql (or explain if refuses)",
        },
        {
            "id": "TC2",
            "question": "Tell me a joke about databases.",
            "description": "Out-of-scope: should explain scope or refuse gracefully",
        },
    ]
    results = []
    for tc in tests:
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tc["question"]},
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1500,
            )
            elapsed = time.time() - start
            msg = resp.choices[0].message
            tc_names = [tc.function.name for tc in msg.tool_calls] if msg.tool_calls else []
            content = msg.content or ""

            if tc["id"] == "TC1":
                # Should call load_skill or explain (appropriate tool use)
                success = bool(msg.tool_calls)
            else:
                # TC2: Should NOT call run_sql for an out-of-scope question
                success = "run_sql" not in tc_names

            results.append({
                "id": tc["id"],
                "description": tc["description"],
                "success": success,
                "tool_calls": tc_names,
                "content_preview": content[:200],
                "elapsed_s": round(elapsed, 2),
            })

        except Exception as e:
            results.append({
                "id": tc["id"],
                "description": tc["description"],
                "success": False,
                "error": str(e),
                "elapsed_s": round(time.time() - start, 2),
            })

    return results


def print_report(model_name: str, sql_results: list[dict], tc_results: list[dict]):
    """Print a formatted benchmark report."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK REPORT: {model_name}")
    print(f"{'='*70}")

    print(f"\n--- SQL Generation ({len(sql_results)} queries) ---")
    sql_passed = sum(1 for r in sql_results if r.get("passed"))
    total_tokens = sum(r.get("completion_tokens", 0) for r in sql_results)
    total_time = sum(r.get("elapsed_s", 0) for r in sql_results)

    for r in sql_results:
        status = "PASS" if r.get("passed") else "FAIL"
        if "error" in r:
            status = "ERR!"
        sql_preview = r.get("sql", "")[:80] if r.get("sql") else "(no SQL)"
        tc_names = r.get("tool_calls", [])
        print(f"  [{status}] {r['id']} ({r['difficulty']}): {r['question']}")
        print(f"         tools: {tc_names}  |  SQL: {sql_preview}")
        if not r.get("passed") and r.get("missing_patterns"):
            print(f"         missing patterns: {r['missing_patterns']}")
        if r.get("raw_content"):
            print(f"         raw: {r['raw_content'][:150]}")

    print(f"  Score: {sql_passed}/{len(sql_results)} "
          f"({100 * sql_passed // len(sql_results) if sql_results else 0}%)")
    print(f"  Total tokens: {total_tokens} | Total time: {total_time:.1f}s")

    print(f"\n--- Tool-Calling Reliability ({len(tc_results)} tests) ---")
    tc_passed = sum(1 for r in tc_results if r.get("success"))
    for r in tc_results:
        status = "PASS" if r.get("success") else "FAIL"
        tc_names = r.get("tool_calls", [])
        print(f"  [{status}] {r['id']}: {r['description']}")
        print(f"         tools called: {tc_names}")
        if r.get("content_preview"):
            print(f"         content: {r['content_preview'][:120]}")

    print(f"  Score: {tc_passed}/{len(tc_results)}")

    print(f"\n--- Summary ---")
    print(f"  Model: {model_name}")
    print(f"  SQL: {sql_passed}/{len(sql_results)}  |  Tool-Calling: {tc_passed}/{len(tc_results)}")
    print()

    return {
        "model": model_name,
        "sql_score": f"{sql_passed}/{len(sql_results)}",
        "tc_score": f"{tc_passed}/{len(tc_results)}",
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-9b-dsv4")
    parser.add_argument("--server", default="http://localhost:8080/v1")
    parser.add_argument("--output", default=None, help="Save JSON report to file")
    args = parser.parse_args()

    client = OpenAI(base_url=args.server, api_key="not-needed")

    # Verify server
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"Server OK — available models: {model_ids}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.server}")
        print(f"  {e}")
        sys.exit(1)

    # Run SQL tests (with skill cards pre-loaded)
    print(f"\nRunning SQL generation tests (skill cards pre-loaded)...")
    sql_results = test_sql_generation(client, args.model, TEST_QUERIES)

    # Run tool-calling tests (bare system prompt, model must call load_skill)
    print(f"Running tool-calling tests...")
    tc_results = test_tool_calling(client, args.model)

    # Print report
    summary = print_report(args.model, sql_results, tc_results)

    # Save JSON report
    if args.output:
        report = {"sql_results": sql_results, "tc_results": tc_results, "summary": summary}
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
