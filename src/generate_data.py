"""
generate_data.py
Generates a realistic synthetic e-commerce / retail transaction dataset
modelled after the UCI Online Retail dataset structure.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

PRODUCTS = [
    ("WHITE HANGING HEART T-LIGHT HOLDER",  2.95, "Gifts"),
    ("REGENCY CAKESTAND 3 TIER",            12.75, "Kitchen"),
    ("JUMBO BAG RED RETROSPOT",              1.65, "Storage"),
    ("PARTY BUNTING",                        4.95, "Party"),
    ("LUNCH BAG RED RETROSPOT",              1.65, "Bags"),
    ("SET OF 3 CAKE TINS PANTRY DESIGN",    4.95, "Kitchen"),
    ("ALARM CLOCK BAKELIKE GREEN",           3.75, "Decor"),
    ("HAND WARMER UNION JACK",               1.85, "Gifts"),
    ("DOORMAT NEW ENGLAND",                  7.95, "Home"),
    ("CERAMIC STORAGE JARS",                 4.25, "Kitchen"),
    ("METAL WALL ART",                       9.95, "Decor"),
    ("SCENTED CANDLE SET",                   6.50, "Home"),
    ("TRAVEL MUG",                           8.95, "Kitchen"),
    ("NOTEBOOK A5",                          3.25, "Stationery"),
    ("WIRELESS CHARGER PAD",                19.99, "Electronics"),
    ("BAMBOO CUTLERY SET",                   5.75, "Kitchen"),
    ("CANVAS TOTE BAG",                      7.50, "Bags"),
    ("SUCCULENT PLANT KIT",                 12.00, "Home"),
    ("PHOTO FRAME MULTI",                    8.50, "Decor"),
    ("DESK ORGANISER",                      11.99, "Stationery"),
]

COUNTRIES = {
    "United Kingdom":  0.60,
    "Germany":         0.08,
    "France":          0.07,
    "Netherlands":     0.05,
    "Australia":       0.04,
    "Spain":           0.04,
    "Belgium":         0.03,
    "Switzerland":     0.03,
    "Portugal":        0.03,
    "Italy":           0.03,
}

N_CUSTOMERS = 4_500
N_TRANSACTIONS = 55_000


def generate_transactions():
    start = datetime(2022, 1, 1)
    end   = datetime(2024, 12, 31)
    date_range_days = (end - start).days

    # Assign customers to countries
    country_keys   = list(COUNTRIES.keys())
    country_weights = list(COUNTRIES.values())
    customer_countries = np.random.choice(country_keys, N_CUSTOMERS, p=country_weights)

    # Seasonal booking weight per day
    def seasonal_weight(dt):
        m = dt.month
        weights = {1:0.7, 2:0.75, 3:0.85, 4:0.9, 5:1.0, 6:1.0,
                   7:1.0, 8:0.95, 9:1.0, 10:1.1, 11:1.3, 12:1.5}
        return weights[m]

    records = []
    invoice_no = 500000

    for _ in range(N_TRANSACTIONS):
        # pick customer
        cust_id = np.random.randint(10000, 10000 + N_CUSTOMERS)
        country = customer_countries[cust_id - 10000]

        # pick date with seasonal weighting
        while True:
            d = start + timedelta(days=int(np.random.randint(0, date_range_days)))
            if np.random.rand() < seasonal_weight(d) / 1.5:
                break

        # pick product
        prod_idx = np.random.randint(0, len(PRODUCTS))
        desc, unit_price, category = PRODUCTS[prod_idx]
        qty = int(np.random.choice([1,2,3,4,6,12], p=[0.35,0.25,0.18,0.10,0.07,0.05]))

        # occasional returns (negative qty)
        if np.random.rand() < 0.02:
            qty = -qty

        invoice_no += np.random.randint(1, 5)
        records.append({
            "InvoiceNo":   str(invoice_no),
            "StockCode":   f"ITEM{prod_idx+1:03d}",
            "Description": desc,
            "Category":    category,
            "Quantity":    qty,
            "InvoiceDate": d.strftime("%Y-%m-%d"),
            "UnitPrice":   unit_price,
            "CustomerID":  cust_id,
            "Country":     country,
        })

    df = pd.DataFrame(records)
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.sort_values("InvoiceDate").reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    print("Generating transactions...")
    df = generate_transactions()
    df.to_csv(out / "transactions.csv", index=False)
    print(f"  Saved {len(df):,} rows → {out}/transactions.csv")
