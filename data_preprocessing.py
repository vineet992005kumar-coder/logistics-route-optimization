"""
Week 2 Task - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Yuva Intern

Simulates a messy logistics dataset (inspired by public e-commerce delivery data such as
the Olist Brazilian E-Commerce dataset), then cleans and preprocesses it with pandas.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

rng = np.random.default_rng(7)


# ------------------------------------------------------------
# 1. Data collection (simulated)
# ------------------------------------------------------------
def simulate_raw_data(n=5000):
    order_time = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 180 * 24 * 60, n), unit="m")
    distance = rng.gamma(2.0, 6.0, n)
    weight = rng.gamma(2.0, 2.5, n)
    transit_hours = 4 + distance * 0.9 + weight * 0.3 + rng.normal(0, 2, n)
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": order_time,
        "delivery_date": order_time + pd.to_timedelta(transit_hours, unit="h"),
        "hub": rng.choice(["Hub A", "Hub B", "Hub C"], n),
        "carrier": rng.choice(["FastShip", "QuickMove", "CityCourier"], n),
        "distance_km": distance,
        "weight_kg": weight,
        "shipping_cost": 20 + distance * 2.2 + weight * 1.5 + rng.normal(0, 5, n),
        "inventory_level": rng.integers(0, 500, n).astype(float),
        "status": rng.choice(["Delivered", "In Transit", "Returned"], n, p=[0.88, 0.08, 0.04]),
    })

    # ---- inject realistic data problems ----
    # missing values
    for i, (col, frac) in enumerate([("distance_km", 0.06), ("weight_kg", 0.04), ("shipping_cost", 0.05),
                                     ("inventory_level", 0.08), ("carrier", 0.03), ("delivery_date", 0.02)]):
        df.loc[df.sample(frac=frac, random_state=100 + i).index, col] = np.nan
    # outliers
    df.loc[df.sample(40, random_state=3).index, "distance_km"] = rng.uniform(400, 900, 40)
    df.loc[df.sample(25, random_state=4).index, "shipping_cost"] = rng.uniform(2000, 5000, 25)
    df.loc[df.sample(10, random_state=5).index, "weight_kg"] = -1.0           # impossible
    # inconsistent text labels
    df.loc[df.sample(200, random_state=6).index, "hub"] = "hub a "
    df.loc[df.sample(150, random_state=8).index, "status"] = "delivered"
    # impossible dates (delivered before ordered)
    bad = df.sample(30, random_state=9).index
    df.loc[bad, "delivery_date"] = df.loc[bad, "order_date"] - pd.Timedelta(days=2)
    # duplicates (same order recorded twice)
    df = pd.concat([df, df.sample(150, random_state=1)], ignore_index=True)
    return df


# ------------------------------------------------------------
# 2. Data quality report
# ------------------------------------------------------------
def quality_report(df, title):
    print(f"\n=== {title} ===")
    print("Rows:", len(df))
    print("Duplicate orders:", df.duplicated(subset="order_id").sum())
    print("Missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])


# ------------------------------------------------------------
# 3. Cleaning
# ------------------------------------------------------------
def clean(df):
    df = df.copy()

    # 3.1 remove exact duplicates
    df = df.drop_duplicates(subset="order_id")

    # 3.2 fix inconsistent text
    df["hub"] = df["hub"].str.strip().str.title()
    df["status"] = df["status"].str.strip().str.title()

    # 3.3 fix impossible values
    df.loc[df["weight_kg"] <= 0, "weight_kg"] = np.nan
    df = df[~(df["delivery_date"] < df["order_date"])]          # delivered before ordered

    # 3.4 missing values
    df["carrier"] = df["carrier"].fillna(df["carrier"].mode()[0])         # categorical -> mode
    for col in ["distance_km", "weight_kg", "shipping_cost"]:
        df[col] = df[col].fillna(df.groupby("hub")[col].transform("median"))  # numeric -> median per hub
    df["inventory_level"] = df["inventory_level"].fillna(df["inventory_level"].median())
    df = df.dropna(subset=["delivery_date"])                    # target-related, cannot guess

    # 3.5 extreme outliers with the IQR rule (3 x IQR) - capped, not deleted
    for col in ["distance_km", "shipping_cost", "weight_kg"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        df[col] = df[col].clip(low, high)
        print(f"Outliers capped in {col}: {n_out}")

    return df


# ------------------------------------------------------------
# 4. Feature engineering and preprocessing
# ------------------------------------------------------------
def preprocess(df):
    df = df.copy()
    df["transit_hours"] = (df["delivery_date"] - df["order_date"]).dt.total_seconds() / 3600
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.dayofweek

    # normalization (0-1) for distance-based models
    num_cols = ["distance_km", "weight_kg", "shipping_cost", "inventory_level"]
    df[[c + "_minmax" for c in num_cols]] = MinMaxScaler().fit_transform(df[num_cols])

    # standardization (mean 0, std 1) for models that assume similar scales
    df[[c + "_z" for c in num_cols]] = StandardScaler().fit_transform(df[num_cols])

    # encode categories
    df = pd.get_dummies(df, columns=["hub", "carrier", "status"], drop_first=True)
    return df


if __name__ == "__main__":
    raw = simulate_raw_data()
    quality_report(raw, "RAW DATA")

    cleaned = clean(raw)
    quality_report(cleaned, "AFTER CLEANING")

    final = preprocess(cleaned)
    print("\nFinal shape:", final.shape)
    final.to_csv("logistics_clean.csv", index=False)
    print("Saved logistics_clean.csv")
