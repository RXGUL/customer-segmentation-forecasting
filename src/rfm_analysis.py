"""
rfm_analysis.py  –  Recency, Frequency, Monetary computation & scoring.
"""

import pandas as pd
import numpy as np


def compute_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Compute RFM values per customer.

    Parameters
    ----------
    df : transactions DataFrame with columns [CustomerID, InvoiceDate, Revenue]
    snapshot_date : reference date (defaults to max date + 1 day)
    """
    df = df[df["Revenue"] > 0].copy()          # exclude returns
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def score_rfm(rfm: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Add R/F/M quintile scores (5 = best) and an RFM_Score composite."""
    rfm = rfm.copy()

    # Recency: lower is better → reverse rank
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=bins,
                               labels=range(bins, 0, -1), duplicates="drop")
    # Frequency & Monetary: higher is better
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=bins,
                               labels=range(1, bins + 1), duplicates="drop")
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=bins,
                               labels=range(1, bins + 1), duplicates="drop")

    for col in ["R_Score", "F_Score", "M_Score"]:
        rfm[col] = rfm[col].astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment_Code"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    return rfm


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Map RFM scores to named customer segments."""

    def segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        score = row["RFM_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r >= 3 and f >= 1 and m >= 3:
            return "Potential Loyalists"
        elif r == 3 and f <= 2:
            return "Promising"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f >= 4:
            return "Cant Lose Them"
        elif r <= 1 and f <= 2 and m <= 2:
            return "Lost"
        elif score <= 6:
            return "Hibernating"
        else:
            return "Need Attention"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm
