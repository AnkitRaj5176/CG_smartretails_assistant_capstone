"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic retail sales CSV for development and testing.

Run from the project root:
    python scripts/generate_sample_data.py

Output: sales_store/retail_records.csv  (5 000 rows by default)
"""

import os
import random
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED: int = 42
NUM_ROWS: int = 5_000
OUTPUT_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "sales_store", "retail_records.csv"
)

PRODUCTS: list[dict] = [
    {"id": "PROD_001", "category": "Electronics",  "base_price": 299.99},
    {"id": "PROD_002", "category": "Electronics",  "base_price": 149.99},
    {"id": "PROD_003", "category": "Clothing",     "base_price": 49.99},
    {"id": "PROD_004", "category": "Clothing",     "base_price": 79.99},
    {"id": "PROD_005", "category": "Groceries",    "base_price": 12.50},
    {"id": "PROD_006", "category": "Groceries",    "base_price": 8.99},
    {"id": "PROD_007", "category": "Furniture",    "base_price": 499.00},
    {"id": "PROD_008", "category": "Sports",       "base_price": 89.99},
    {"id": "PROD_009", "category": "Beauty",       "base_price": 34.99},
    {"id": "PROD_010", "category": "Toys",         "base_price": 24.99},
    {"id": "PROD_011", "category": "Books",        "base_price": 19.99},
    {"id": "PROD_012", "category": "Automotive",   "base_price": 129.99},
    {"id": "PROD_013", "category": "Health",       "base_price": 44.99},
    {"id": "PROD_014", "category": "Electronics",  "base_price": 599.99},
    {"id": "PROD_015", "category": "Sports",       "base_price": 199.99},
]

STORES: list[str] = ["STORE_A", "STORE_B", "STORE_C", "STORE_D",
                     "STORE_E", "STORE_F", "STORE_G", "STORE_H"]

REGIONS: list[str] = ["North", "South", "East", "West", "Central",
                      "Northeast", "Northwest", "Southeast"]

START_DATE: date = date(2023, 1, 1)
END_DATE: date = date(2024, 12, 31)
DATE_RANGE: int = (END_DATE - START_DATE).days


def _random_date() -> date:
    return START_DATE + timedelta(days=random.randint(0, DATE_RANGE))


def _seasonal_multiplier(sale_date: date) -> float:
    month = sale_date.month
    if month in (11, 12):
        return 1.6
    if month in (6, 7, 8):
        return 1.2
    if month in (1, 2):
        return 0.8
    return 1.0


def generate_records(num_rows: int) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    rows = []
    for _ in range(num_rows):
        product = random.choice(PRODUCTS)
        store_id = random.choice(STORES)
        region = random.choice(REGIONS)
        sale_date = _random_date()

        price = round(product["base_price"] * np.random.uniform(0.9, 1.1), 2)
        discount = round(
            np.random.choice(
                [0, 5, 10, 15, 20, 25, 30, 40],
                p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03],
            ), 2,
        )

        base_units = max(1, int(300 / price * 10))
        seasonal = _seasonal_multiplier(sale_date)
        discount_boost = 1 + discount / 200
        units_sold = max(
            0,
            int(np.random.poisson(base_units * seasonal * discount_boost)
                + np.random.randint(-2, 3)),
        )

        # Inject ~3% anomalies
        if random.random() < 0.015:
            units_sold = int(units_sold * random.uniform(4, 8))
        elif random.random() < 0.015:
            units_sold = 0

        revenue = round(price * units_sold * (1 - discount / 100), 2)

        rows.append({
            "product_id": product["id"],
            "category": product["category"],
            "region": region,
            "store_id": store_id,
            "date": sale_date.strftime("%Y-%m-%d"),
            "price": price,
            "discount": discount,
            "units_sold": units_sold,
            "revenue": revenue,
        })

    return pd.DataFrame(rows)


def main() -> None:
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {NUM_ROWS} synthetic retail records ...")
    df = generate_records(NUM_ROWS)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved  : {os.path.abspath(OUTPUT_PATH)}")
    print(f"Shape  : {df.shape}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
