import pandas as pd
import numpy as np
import os

# Create folder if it doesn't exist
os.makedirs("./test_data", exist_ok=True)

# 1. GENERATE CLEAN DATASET (with warnings: insufficient history, no price variation)
rows = []
start_date = pd.to_datetime("2026-06-01")

# SKU-001: Clean with price variation & competitor price & promo
np.random.seed(42)
for i in range(30):
    date_str = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    price = float(np.random.choice([10.0, 12.0, 9.0, 11.0]))
    comp_price = float(price + np.random.uniform(-1.0, 1.5))
    promo = int(np.random.choice([0, 1], p=[0.7, 0.3]))
    # Demand function: log(units) = 5 - 1.5 * log(price) + 0.4 * log(comp) + 0.3 * promo + noise
    log_units = 5 - 1.5 * np.log(price) + 0.4 * np.log(comp_price) + 0.3 * promo + np.random.normal(0, 0.1)
    units_sold = int(np.exp(log_units))
    
    rows.append({
        "product_id": "SKU-001",
        "date": date_str,
        "price": price,
        "quantity": units_sold,
        "competitor_price": comp_price,
        "promo": promo,
        "unit_cost": 5.0,
        "category": "Electronics"
    })

# SKU-002: Clean, simple variation, no promo/comp
for i in range(25):
    date_str = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    price = float(np.random.choice([20.0, 24.0, 18.0]))
    units_sold = int(120 - 3 * price + np.random.normal(0, 2))
    rows.append({
        "product_id": "SKU-002",
        "date": date_str,
        "price": price,
        "quantity": units_sold,
        "competitor_price": None,
        "promo": None,
        "unit_cost": 10.0,
        "category": "Apparel"
    })

# SKU-003: Warnings - No price variation (constant price 15.0)
for i in range(20):
    date_str = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    rows.append({
        "product_id": "SKU-003",
        "date": date_str,
        "price": 15.0,
        "quantity": 12,
        "competitor_price": 14.5,
        "promo": 0,
        "unit_cost": 7.5,
        "category": "Apparel"
    })

# SKU-004: Warnings - Insufficient history (only 5 rows)
for i in range(5):
    date_str = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    rows.append({
        "product_id": "SKU-004",
        "date": date_str,
        "price": float(np.random.choice([30.0, 35.0])),
        "quantity": int(10 + np.random.normal(0, 1)),
        "competitor_price": 32.0,
        "promo": 0,
        "unit_cost": 15.0,
        "category": "Office"
    })

df_clean = pd.DataFrame(rows)
df_clean.to_csv("./test_data/clean_dataset.csv", index=False)
print("Saved clean_dataset.csv with", len(df_clean), "rows")

# 2. GENERATE INVALID DATASET (blocking errors)
rows_invalid = []
# Missing required columns (e.g. missing units_sold/quantity)
for i in range(10):
    date_str = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    rows_invalid.append({
        "product_id": "SKU-BAD",
        "date": date_str,
        "price": 10.0,
        # units_sold / quantity is missing!
        "competitor_price": 11.0
    })

df_invalid = pd.DataFrame(rows_invalid)
df_invalid.to_csv("./test_data/missing_columns_dataset.csv", index=False)
print("Saved missing_columns_dataset.csv")

# 3. GENERATE INVALID DATASET (negative prices / invalid types)
rows_neg = []
for i in range(15):
    date_str = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    # Row 5 has a negative price
    price = -5.0 if i == 5 else 12.0
    # Row 8 has non-numeric units
    qty = "invalid_text" if i == 8 else 45
    
    rows_neg.append({
        "product_id": "SKU-NEG",
        "date": date_str,
        "price": price,
        "quantity": qty,
        "category": "Groceries"
    })
df_neg = pd.DataFrame(rows_neg)
df_neg.to_csv("./test_data/invalid_values_dataset.csv", index=False)
print("Saved invalid_values_dataset.csv")
