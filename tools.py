"""OpenAI function-calling tool definitions for the NL2SQL agent."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load a domain skill card to get relevant schema context and example queries. "
                "Available skills: loan-analysis, customer-insights, transaction-analysis, "
                "account-overview, duckdb-rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": [
                            "loan-analysis",
                            "customer-insights",
                            "transaction-analysis",
                            "account-overview",
                            "duckdb-rules",
                        ],
                        "description": "The name of the skill to load.",
                    }
                },
                "required": ["skill"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a SQL query against the DuckDB database. "
                "Returns the result rows or an error message if the query fails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The DuckDB SQL query to execute.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schema_check",
            "description": (
                "Look up column names and types for a given table. "
                "Use this when run_sql returns a column-not-found error to verify column names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The table name to look up (customers, accounts, transactions, loans).",
                    }
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain",
            "description": (
                "Provide a natural-language summary to the user. "
                "Call this as the final step after executing SQL or when you cannot answer. "
                "If run_sql returned no rows, tell the user no data matched. "
                "If the question is outside the financial domain, explain your scope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The natural-language summary to show the user.",
                    }
                },
                "required": ["text"],
            },
        },
    },
]
