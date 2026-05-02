"""Domain skill cards and schema lookup for the NL2SQL agent."""

from pathlib import Path
import duckdb
from config import DB_PATH


DUCKDB_RULES = """\
DuckDB syntax rules:
- No :: type casts. Use CAST(expr AS type) instead.
- No ILIKE. Use LOWER(col) LIKE '%pattern%' instead.
- No pg_catalog references.
- Date ranges: col >= 'YYYY-MM-DD' AND col < 'YYYY-MM-DD'
- For month extraction: EXTRACT(MONTH FROM date_col)
- For year extraction: EXTRACT(YEAR FROM date_col)
- Do NOT use strftime(). Prefer range comparisons for date filtering.
- Use date_trunc('month', date_col) to group by month.
"""

LOAN_ANALYSIS = """\
Relevant tables for loan questions:

loans
  loan_id INTEGER PRIMARY KEY
  customer_id INTEGER -> customers(customer_id)
  loan_type VARCHAR          -- Home Loan, Car Loan, Personal Loan, Education Loan, Business Loan
  principal DOUBLE           -- original loan amount
  interest_rate DOUBLE       -- annual rate (e.g. 4.5 means 4.5%)
  remaining_balance DOUBLE   -- amount still owed
  start_date DATE
  term_months INTEGER

customers
  customer_id INTEGER PRIMARY KEY
  name VARCHAR
  segment VARCHAR            -- Retail, Priority, Private, SME
  age INTEGER
  city VARCHAR
  join_date DATE

Example queries (adapt column names and values to the user's question):

-- Top 5 customers by remaining loan balance
SELECT c.name, SUM(l.remaining_balance) AS total_owed
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
GROUP BY c.name
ORDER BY total_owed DESC
LIMIT 5

-- Average interest rate by loan type
SELECT l.loan_type, ROUND(AVG(l.interest_rate), 2) AS avg_rate, COUNT(*) AS total_loans
FROM loans l
GROUP BY l.loan_type
ORDER BY avg_rate DESC

-- Total outstanding loan balance by customer segment
SELECT c.segment, SUM(l.remaining_balance) AS total_outstanding, COUNT(*) AS loan_count
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
GROUP BY c.segment
ORDER BY total_outstanding DESC

-- Home loans starting in Q1 2025
SELECT c.name, l.principal, l.interest_rate, l.start_date
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
WHERE l.loan_type = 'Home Loan'
  AND l.start_date >= '2025-01-01' AND l.start_date < '2025-04-01'
ORDER BY l.start_date
"""

CUSTOMER_INSIGHTS = """\
Relevant tables for customer questions:

customers
  customer_id INTEGER PRIMARY KEY
  name VARCHAR
  segment VARCHAR            -- Retail, Priority, Private, SME
  age INTEGER
  city VARCHAR
  join_date DATE

accounts
  account_id INTEGER PRIMARY KEY
  customer_id INTEGER -> customers(customer_id)
  account_type VARCHAR       -- Savings, Current, Fixed Deposit, Investment
  balance DOUBLE
  currency VARCHAR           -- MYR, USD, SGD
  opened_date DATE

Example queries:

-- Customer count by segment
SELECT segment, COUNT(*) AS customer_count
FROM customers
GROUP BY segment
ORDER BY customer_count DESC

-- Average age by city (top 5)
SELECT city, ROUND(AVG(age), 1) AS avg_age, COUNT(*) AS count
FROM customers
GROUP BY city
ORDER BY avg_age DESC
LIMIT 5

-- Customers who joined in 2024
SELECT name, city, segment, join_date
FROM customers
WHERE join_date >= '2024-01-01' AND join_date < '2025-01-01'
ORDER BY join_date

-- Total balance per customer (customers with multiple accounts)
SELECT c.name, SUM(a.balance) AS total_balance, COUNT(a.account_id) AS account_count
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
GROUP BY c.name
ORDER BY total_balance DESC
LIMIT 10
"""

TRANSACTION_ANALYSIS = """\
Relevant tables for transaction questions:

transactions
  transaction_id INTEGER PRIMARY KEY
  account_id INTEGER -> accounts(account_id)
  customer_id INTEGER
  date DATE
  amount DOUBLE              -- positive = credit, negative = debit
  category VARCHAR
  description VARCHAR

accounts
  account_id INTEGER PRIMARY KEY
  customer_id INTEGER -> customers(customer_id)
  account_type VARCHAR
  balance DOUBLE
  currency VARCHAR
  opened_date DATE

customers
  customer_id INTEGER PRIMARY KEY
  name VARCHAR
  segment VARCHAR
  age INTEGER
  city VARCHAR
  join_date DATE

Example queries:

-- Total transaction amount by category
SELECT t.category, SUM(t.amount) AS total, COUNT(*) AS count
FROM transactions t
GROUP BY t.category
ORDER BY total DESC

-- Monthly spending for a specific customer (use name from customers table)
SELECT c.name, DATE_TRUNC('month', t.date) AS month, SUM(t.amount) AS net_amount
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE c.name = 'Alice Tan'
GROUP BY c.name, DATE_TRUNC('month', t.date)
ORDER BY month

-- Top 10 largest individual transactions (by absolute amount)
SELECT t.transaction_id, c.name, t.date, t.amount, t.category
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
ORDER BY ABS(t.amount) DESC
LIMIT 10

-- Transactions in March 2025
SELECT c.name, t.date, t.amount, t.category
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE t.date >= '2025-03-01' AND t.date < '2025-04-01'
ORDER BY t.date
"""

ACCOUNT_OVERVIEW = """\
Relevant tables for account questions:

accounts
  account_id INTEGER PRIMARY KEY
  customer_id INTEGER -> customers(customer_id)
  account_type VARCHAR       -- Savings, Current, Fixed Deposit, Investment
  balance DOUBLE
  currency VARCHAR           -- MYR, USD, SGD
  opened_date DATE

customers
  customer_id INTEGER PRIMARY KEY
  name VARCHAR
  segment VARCHAR
  age INTEGER
  city VARCHAR
  join_date DATE

Example queries:

-- Total balance by account type
SELECT a.account_type, SUM(a.balance) AS total_balance, COUNT(*) AS account_count
FROM accounts a
GROUP BY a.account_type
ORDER BY total_balance DESC

-- Accounts with balance above 100,000
SELECT a.account_id, c.name, a.account_type, a.balance, a.currency
FROM accounts a
JOIN customers c ON a.customer_id = c.customer_id
WHERE a.balance > 100000
ORDER BY a.balance DESC

-- Accounts by currency
SELECT a.currency, COUNT(*) AS account_count, SUM(a.balance) AS total_balance
FROM accounts a
GROUP BY a.currency
ORDER BY total_balance DESC

-- Average balance per customer segment
SELECT c.segment, ROUND(AVG(a.balance), 2) AS avg_balance, COUNT(*) AS account_count
FROM accounts a
JOIN customers c ON a.customer_id = c.customer_id
GROUP BY c.segment
ORDER BY avg_balance DESC
"""


SKILLS = {
    "loan-analysis": LOAN_ANALYSIS,
    "customer-insights": CUSTOMER_INSIGHTS,
    "transaction-analysis": TRANSACTION_ANALYSIS,
    "account-overview": ACCOUNT_OVERVIEW,
    "duckdb-rules": DUCKDB_RULES,
}


SCHEMA_MAP = {
    "customers": [
        ("customer_id", "INTEGER"),
        ("name", "VARCHAR"),
        ("segment", "VARCHAR"),
        ("age", "INTEGER"),
        ("city", "VARCHAR"),
        ("join_date", "DATE"),
    ],
    "accounts": [
        ("account_id", "INTEGER"),
        ("customer_id", "INTEGER"),
        ("account_type", "VARCHAR"),
        ("balance", "DOUBLE"),
        ("currency", "VARCHAR"),
        ("opened_date", "DATE"),
    ],
    "transactions": [
        ("transaction_id", "INTEGER"),
        ("account_id", "INTEGER"),
        ("customer_id", "INTEGER"),
        ("date", "DATE"),
        ("amount", "DOUBLE"),
        ("category", "VARCHAR"),
        ("description", "VARCHAR"),
    ],
    "loans": [
        ("loan_id", "INTEGER"),
        ("customer_id", "INTEGER"),
        ("loan_type", "VARCHAR"),
        ("principal", "DOUBLE"),
        ("interest_rate", "DOUBLE"),
        ("remaining_balance", "DOUBLE"),
        ("start_date", "DATE"),
        ("term_months", "INTEGER"),
    ],
}


def schema_lookup(table: str) -> str:
    """Return column names and types for a table. Case-insensitive match."""
    lower = table.lower().strip()
    for name, cols in SCHEMA_MAP.items():
        if name.lower() == lower:
            return "\n".join(f"  {col} {typ}" for col, typ in cols)
    available = ", ".join(SCHEMA_MAP.keys())
    return f"Table '{table}' not found. Available tables: {available}"


def get_row_counts() -> dict:
    """Return row counts for all tables. Used for schema verification."""
    conn = duckdb.connect(DB_PATH)
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    counts = {}
    for (t,) in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        counts[t] = cnt
    conn.close()
    return counts
