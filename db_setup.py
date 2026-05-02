"""Create the sample financial database in DuckDB."""

import random
from datetime import datetime, timedelta

import duckdb
import pandas as pd

from config import DB_PATH

SEGMENTS = ["Retail", "Priority", "Private", "SME"]
CITIES = [
    "Kuala Lumpur", "Penang", "Johor Bahru", "Kota Kinabalu",
    "Kuching", "Ipoh", "Melaka", "Shah Alam",
]
ACCOUNT_TYPES = ["Savings", "Current", "Fixed Deposit", "Investment"]
CURRENCIES = ["MYR", "USD", "SGD"]
TXN_CATEGORIES = [
    "Food & Dining", "Transport", "Shopping", "Utilities",
    "Entertainment", "Salary", "Investment", "Transfer",
    "Insurance", "Loan Payment", "Rental", "Healthcare",
]
LOAN_TYPES = ["Home Loan", "Car Loan", "Personal Loan", "Education Loan", "Business Loan"]

NAMES = [
    "Alice Tan", "Bob Lee", "Charlie Ng", "Diana Wong", "Ethan Lim",
    "Fiona Chen", "George Goh", "Hannah Teo", "Ivan Chua", "Jessica Koh",
    "Kevin Foo", "Laura Heng", "Marcus Yap", "Natalie Sim", "Oscar Phua",
    "Patricia Yeo", "Quentin Loh", "Rachel Wee", "Samuel Gan", "Tiffany Ho",
    "Uma Raj", "Victor Chew", "Wendy Soh", "Xavier Png", "Yvonne Ong",
    "Zachary Tay", "Amanda Liew", "Benjamin Chong", "Catherine Quah",
    "Daniel Seah", "Emily Lau", "Franklin Kwan", "Grace Pang",
    "Henry Tong", "Isabel Chan", "Jackie Yip", "Karen Beh",
    "Leonard Kuah", "Michelle Ngo", "Nicholas Fong", "Olivia Hiew",
    "Patrick Chow", "Queenie Lam", "Richard Teng", "Sharon Ang",
    "Thomas Boon", "Ursula Khoo", "Vincent Mok", "Winnie Law", "Xander Chin",
]

CREDIT_CATEGORIES = {"Salary", "Investment", "Transfer"}


def _rand_date(start: datetime, days: int) -> str:
    return (start + timedelta(days=random.randint(0, days))).strftime("%Y-%m-%d")


def create_database() -> None:
    conn = duckdb.connect(DB_PATH)

    customers = pd.DataFrame({
        "customer_id": range(1, 51),
        "name": NAMES,
        "segment": random.choices(SEGMENTS, weights=[0.40, 0.30, 0.10, 0.20], k=50),
        "age": [random.randint(21, 75) for _ in range(50)],
        "city": random.choices(CITIES, k=50),
        "join_date": [_rand_date(datetime(2023, 1, 1), 1095) for _ in range(50)],
    })

    accounts = []
    acc_id = 1
    for cust_id in range(1, 51):
        for _ in range(random.randint(1, 3)):
            atype = random.choice(ACCOUNT_TYPES)
            balance = round(random.uniform(10000, 1000000), 2) if atype == "Fixed Deposit" else round(random.uniform(500, 500000), 2)
            accounts.append({
                "account_id": acc_id,
                "customer_id": cust_id,
                "account_type": atype,
                "balance": balance,
                "currency": random.choice(CURRENCIES),
                "opened_date": _rand_date(datetime(2023, 1, 1), 1000),
            })
            acc_id += 1
    accounts_df = pd.DataFrame(accounts)

    txns = []
    txn_id = 1
    for acc in accounts:
        base_date = datetime.strptime(acc["opened_date"], "%Y-%m-%d")
        for _ in range(random.randint(10, 40)):
            cat = random.choice(TXN_CATEGORIES)
            is_credit = cat in CREDIT_CATEGORIES
            amount = round(random.uniform(1000, 25000), 2) if is_credit else -round(random.uniform(10, 5000), 2)
            txn_date = base_date + timedelta(days=random.randint(1, 365))
            if txn_date > datetime(2025, 12, 31):
                txn_date = datetime(2025, 12, 31) - timedelta(days=random.randint(1, 90))
            txns.append({
                "transaction_id": txn_id,
                "account_id": acc["account_id"],
                "customer_id": acc["customer_id"],
                "date": txn_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "category": cat,
                "description": f"{cat} - Txn #{txn_id}",
            })
            txn_id += 1
    txns_df = pd.DataFrame(txns)

    loans = []
    loan_id = 1
    for cust_id in range(1, 51):
        if random.random() < 0.4:
            continue
        for _ in range(random.randint(1, 2)):
            ltype = random.choice(LOAN_TYPES)
            principal = round(random.uniform(200000, 2000000), 2) if ltype == "Home Loan" else round(random.uniform(5000, 500000), 2)
            rate = round(random.uniform(2.5, 8.5), 2)
            remaining = round(principal * random.uniform(0.2, 0.95), 2)
            loans.append({
                "loan_id": loan_id,
                "customer_id": cust_id,
                "loan_type": ltype,
                "principal": principal,
                "interest_rate": rate,
                "remaining_balance": remaining,
                "start_date": _rand_date(datetime(2023, 1, 1), 1000),
                "term_months": random.choice([12, 24, 36, 60, 120, 240, 360]),
            })
            loan_id += 1
    loans_df = pd.DataFrame(loans)

    for table in ["transactions", "accounts", "loans", "customers"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.execute("CREATE TABLE customers AS SELECT * FROM customers")
    conn.execute("CREATE TABLE accounts AS SELECT * FROM accounts_df")
    conn.execute("CREATE TABLE transactions AS SELECT * FROM txns_df")
    conn.execute("CREATE TABLE loans AS SELECT * FROM loans_df")

    date_cols = [
        ("transactions", "date"),
        ("customers", "join_date"),
        ("accounts", "opened_date"),
        ("loans", "start_date"),
    ]
    for tbl, col in date_cols:
        conn.execute(f"ALTER TABLE {tbl} ALTER {col} TYPE DATE USING {col}::DATE")

    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    print("Database created.")
    for (t,) in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {cnt} rows")

    conn.close()
    print(f"\nSaved to {DB_PATH}")


if __name__ == "__main__":
    create_database()
