"""
data_generator.py
-----------------
Generates a realistic synthetic e-commerce transaction dataset.
Produces ~50,000 orders across 2,500 customers over 3 years.
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "transactions.csv")


def generate_transactions(
    n_customers: int = 2500,
    start_date: str = "2022-01-01",
    end_date:   str = "2024-12-31",
    seed:       int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic e-commerce transactions with realistic purchasing patterns.

    Customer segments are baked in (hidden ground truth for clustering validation):
        - Champions    : very frequent, high spend, very recent
        - Loyal        : frequent, medium-high spend
        - At Risk       : previously good, fading activity
        - Hibernating  : haven't purchased in a long time
        - New          : recently acquired, low history

    Returns
    -------
    pd.DataFrame with columns:
        order_id, customer_id, order_date, quantity, unit_price,
        total_amount, product_category, country
    """
    rng = np.random.default_rng(seed)

    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    total_days = (end_ts - start_ts).days

    # ── Customer segment profiles ─────────────────────────────────────────────
    segments = {
        "Champions":   {"frac": 0.10, "n_orders_mean": 35, "spend_mean": 320, "recent_bias": 0.85},
        "Loyal":       {"frac": 0.20, "n_orders_mean": 18, "spend_mean": 180, "recent_bias": 0.70},
        "Promising":   {"frac": 0.15, "n_orders_mean": 10, "spend_mean": 130, "recent_bias": 0.65},
        "At Risk":     {"frac": 0.20, "n_orders_mean":  8, "spend_mean": 110, "recent_bias": 0.25},
        "Hibernating": {"frac": 0.20, "n_orders_mean":  3, "spend_mean":  80, "recent_bias": 0.05},
        "New":         {"frac": 0.15, "n_orders_mean":  2, "spend_mean": 100, "recent_bias": 0.90},
    }

    categories = ["Electronics", "Clothing", "Home & Garden",
                  "Books", "Sports", "Beauty", "Toys", "Food"]
    cat_weights = [0.20, 0.18, 0.15, 0.10, 0.12, 0.10, 0.08, 0.07]

    countries = ["USA", "UK", "Germany", "France", "Canada",
                 "Australia", "India", "Brazil"]
    country_weights = [0.35, 0.15, 0.12, 0.10, 0.10, 0.08, 0.05, 0.05]

    records = []
    order_id = 1
    cust_id  = 1

    for seg_name, seg in segments.items():
        n_seg = int(n_customers * seg["frac"])
        for _ in range(n_seg):
            n_orders = max(1, int(rng.normal(seg["n_orders_mean"],
                                             seg["n_orders_mean"] * 0.4)))
            # Bias order dates toward recent (for active segments) or older (dormant)
            recent_bias = seg["recent_bias"]
            # Beta distribution: high alpha+low beta → right skew (recent)
            alpha = 1 + recent_bias * 4
            beta  = 1 + (1 - recent_bias) * 4
            day_fracs = rng.beta(alpha, beta, size=n_orders)
            order_days = (day_fracs * total_days).astype(int)
            order_dates = [start_ts + pd.Timedelta(days=int(d)) for d in order_days]

            for odate in order_dates:
                category  = rng.choice(categories, p=cat_weights)
                country   = rng.choice(countries,  p=country_weights)
                quantity  = int(rng.choice([1, 2, 3, 4, 5], p=[0.50, 0.25, 0.13, 0.07, 0.05]))

                # Unit price varies by category
                cat_base = {
                    "Electronics": 250, "Clothing": 60, "Home & Garden": 80,
                    "Books": 20, "Sports": 70, "Beauty": 40, "Toys": 35, "Food": 25,
                }
                base = cat_base.get(category, 50)
                spend_scale = seg["spend_mean"] / 180  # normalise against "Loyal"
                unit_price = max(5.0, round(
                    base * spend_scale * rng.uniform(0.7, 1.3), 2
                ))
                total_amount = round(unit_price * quantity, 2)

                records.append({
                    "order_id":        order_id,
                    "customer_id":     cust_id,
                    "order_date":      odate.strftime("%Y-%m-%d"),
                    "quantity":        quantity,
                    "unit_price":      unit_price,
                    "total_amount":    total_amount,
                    "product_category": category,
                    "country":         country,
                    "true_segment":    seg_name,  # ground truth for validation
                })
                order_id += 1
            cust_id += 1

    df = pd.DataFrame(records)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df.sort_values("order_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[data_generator] Generated {len(df):,} transactions | "
          f"{df['customer_id'].nunique():,} customers | "
          f"{df['order_date'].min().date()} → {df['order_date'].max().date()}")
    return df


def save_raw(df: pd.DataFrame, path: str = RAW_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_generator] Saved → {path}")
    return path


def load_or_generate(path: str = RAW_PATH, **kwargs) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["order_date"])
        print(f"[data_generator] Loaded existing data: {len(df):,} rows")
        return df
    df = generate_transactions(**kwargs)
    save_raw(df, path)
    return df
