"""Benchmark a model for NL2SQL agent: SQL quality + tool-calling reliability.

Usage:
    python benchmark_model.py [--model MODEL_NAME] [--server URL]

Compares the model against the calibrated test suite.
"""

import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# --- Test Queries (question, expected SQL pattern, difficulty) ---
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
        "sql_must_contain": ["JOIN", "loans", "customers", "remaining_balance", ">", "100000"],
        "difficulty": "medium",
        "skill": "loan-analysis",
    },
    {
        "id": "Q4",
        "question": "What is the average transaction amount per category for Retail segment customers?",
        "sql_must_contain": ["AVG", "category", "GROUP BY", "segment", "Retail", "JOIN", "transactions"],
        "difficulty": "hard",
        "skill": "transaction-analysis",
    },
    {
        "id": "Q5",
        "question": "Find customers whose total spending exceeds 100,000 MYR. Show name and total spent.",
        "sql_must_contain": ["SUM", "GROUP BY", "HAVING", "100000", "JOIN"],
        "difficulty": "hard",
        "skill": "transaction-analysis",
    },
]

# --- Tool-Calling Test Prompts ---
TOOL_CALL_TESTS = [
    {
        "id": "TC1",
        "prompt": "How many customers are in Kuala Lumpur?",
        "expects": "load_skill",
        "description": "Basic: should call load_skill first",
    },
    {
        "id": "TC2",
        "prompt": "Tell me a joke.",
        "expects": "refuse_or_explain",
        "description": "Out-of-scope: should refuse or explain scope",
    },
]

SYSTEM_PROMPT = """\
You are a financial database agent. You can ONLY respond using function calls.
NEVER answer a question directly — always use the functions below.

Format for function calls:

<tool_call>
{"arguments": {<args>}, "name": "<function_name>"}
</tool_call>

After each call, you will receive a <tool_response> with the result.

Functions:

load_skill(skill)
  Load domain context before querying.
  skill: "loan-analysis" | "customer-insights" | "transaction-analysis" | "account-overview"

run_sql(sql)
  Execute DuckDB SQL. Returns rows or an error inside <tool_response>.

schema_check(table)
  Look up columns for a table. Use when run_sql fails.

explain(text)
  Summarize results for the user. Final step — conversation ends after this.

Required workflow for EVERY question:
1. load_skill  →  get the right domain context
2. run_sql     →  execute the query
3. explain     →  summarize results

Rules:
- ALWAYS call load_skill first.
- If run_sql returns an error, call schema_check, then retry run_sql.
- If 0 rows returned, explain that no data matched.
- NEVER output SQL or explanations as plain text — use run_sql and explain.
"""


def check_sql_pattern(sql: str, must_contain: list[str]) -> tuple[bool, list[str]]:
    """Check if SQL contains all required patterns (case-insensitive)."""
    sql_upper = sql.upper()
    missing = [p for p in must_contain if p.upper() not in sql_upper]
    return len(missing) == 0, missing


def test_sql_generation(client: OpenAI, model: str, queries: list[dict]) -> list[dict]:
    """Test SQL generation by sending a single-turn request (no tool loop)."""
    results = []
    for q in queries:
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q["question"]},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            elapsed = time.time() - start
            content = resp.choices[0].message.content or ""
            tokens = resp.usage.completion_tokens if resp.usage else 0

            # Extract SQL if present
            sql = ""
            if "<tool_call>" in content:
                # Try to parse tool call
                import re
                m = re.search(r'"sql"\s*:\s*"((?:\\.|[^"\\])*)"', content)
                if m:
                    sql = m.group(1).encode().decode("unicode_escape")
            elif "SELECT" in content.upper():
                # Model might output SQL directly (not ideal but extractable)
                import re
                m = re.search(r'(SELECT\s+.+?(?:;|$))', content, re.DOTALL | re.IGNORECASE)
                if m:
                    sql = m.group(1).strip()

            passed, missing = check_sql_pattern(sql, q["sql_must_contain"])
            results.append({
                "id": q["id"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "passed": passed,
                "missing_patterns": missing,
                "sql_found": bool(sql),
                "raw_response": content[:300],
                "elapsed_s": round(elapsed, 2),
                "completion_tokens": tokens,
            })
        except Exception as e:
            results.append({
                "id": q["id"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "passed": False,
                "error": str(e),
                "elapsed_s": time.time() - start,
            })
    return results


def test_tool_calling(client: OpenAI, model: str, tests: list[dict]) -> list[dict]:
    """Test whether the model emits proper tool calls."""
    results = []
    for tc in tests:
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tc["prompt"]},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            elapsed = time.time() - start
            content = resp.choices[0].message.content or ""

            has_tool_call = "<tool_call>" in content
            has_load_skill = "load_skill" in content
            has_explain = "explain" in content
            has_run_sql = "run_sql" in content

            # Determine if response matches expectations
            if tc["expects"] == "load_skill":
                success = has_tool_call and (has_load_skill or has_run_sql or has_explain)
            elif tc["expects"] == "refuse_or_explain":
                success = not has_run_sql  # Should not try to run SQL for out-of-scope
            else:
                success = has_tool_call

            results.append({
                "id": tc["id"],
                "description": tc["description"],
                "prompt": tc["prompt"],
                "success": success,
                "has_tool_call": has_tool_call,
                "has_load_skill": has_load_skill,
                "raw_snippet": content[:200],
                "elapsed_s": round(elapsed, 2),
            })
        except Exception as e:
            results.append({
                "id": tc["id"],
                "description": tc["description"],
                "success": False,
                "error": str(e),
                "elapsed_s": time.time() - start,
            })
    return results


def print_report(model_name: str, sql_results: list[dict], tc_results: list[dict]):
    """Print a formatted benchmark report."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK REPORT: {model_name}")
    print(f"{'='*70}")

    # SQL Quality
    print(f"\n--- SQL Generation ({len(sql_results)} queries) ---")
    sql_passed = sum(1 for r in sql_results if r.get("passed"))
    sql_total = len(sql_results)
    total_tokens = sum(r.get("completion_tokens", 0) for r in sql_results)
    total_time = sum(r.get("elapsed_s", 0) for r in sql_results)

    for r in sql_results:
        status = "PASS" if r.get("passed") else "FAIL"
        if "error" in r:
            status = "ERR!"
        print(f"  [{status}] {r['id']} ({r['difficulty']}): {r['question']}")
        if not r.get("passed") and "missing_patterns" in r:
            print(f"         Missing: {r['missing_patterns']}")
        if "error" in r:
            print(f"         Error: {r['error'][:120]}")
        if not r.get("sql_found", True):
            print(f"         No SQL found in response")
    print(f"  Score: {sql_passed}/{sql_total} ({100*sql_passed//sql_total if sql_total else 0}%)")
    print(f"  Total tokens: {total_tokens} | Total time: {total_time:.1f}s")

    # Tool-Calling
    print(f"\n--- Tool-Calling Reliability ({len(tc_results)} tests) ---")
    tc_passed = sum(1 for r in tc_results if r.get("success"))
    for r in tc_results:
        status = "PASS" if r.get("success") else "FAIL"
        print(f"  [{status}] {r['id']}: {r['description']}")
        if not r.get("has_tool_call", False):
            print(f"         No <tool_call> in response")
        if "error" in r:
            print(f"         Error: {r['error'][:120]}")
    print(f"  Score: {tc_passed}/{len(tc_results)}")

    print(f"\n--- Summary ---")
    print(f"  Model: {model_name}")
    print(f"  SQL: {sql_passed}/{sql_total} | Tool-Calling: {tc_passed}/{len(tc_results)}")
    print()

    return {
        "model": model_name,
        "sql_score": f"{sql_passed}/{sql_total}",
        "tc_score": f"{tc_passed}/{len(tc_results)}",
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Hermes-2-Pro-Llama-3-8B")
    parser.add_argument("--server", default="http://localhost:8080/v1")
    parser.add_argument("--output", default=None, help="Save JSON report to file")
    args = parser.parse_args()

    client = OpenAI(base_url=args.server, api_key="not-needed")

    # Verify server
    try:
        models = client.models.list()
        print(f"Server OK — available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.server}")
        print(f"  {e}")
        sys.exit(1)

    # Run SQL tests
    print(f"\nRunning SQL generation tests...")
    sql_results = test_sql_generation(client, args.model, TEST_QUERIES)

    # Run tool-calling tests
    print(f"Running tool-calling tests...")
    tc_results = test_tool_calling(client, args.model, TOOL_CALL_TESTS)

    # Print report
    summary = print_report(args.model, sql_results, tc_results)

    # Save
    if args.output:
        report = {"sql_results": sql_results, "tc_results": tc_results, "summary": summary}
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
