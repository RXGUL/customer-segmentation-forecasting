"""
forecasting.py  –  Sales forecasting using Facebook Prophet.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


def prepare_prophet_df(transactions: pd.DataFrame,
                        freq: str = "W") -> pd.DataFrame:
    """
    Aggregate transactions to weekly or monthly revenue,
    returning a Prophet-compatible df with columns [ds, y].
    """
    ts = (
        transactions[transactions["Revenue"] > 0]
        .groupby(pd.Grouper(key="InvoiceDate", freq=freq))["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDate": "ds", "Revenue": "y"})
    )
    ts = ts[ts["y"] > 0]
    return ts


def fit_prophet(ts: pd.DataFrame, periods: int = 26,
                freq: str = "W") -> tuple:
    """
    Fit a Prophet model with yearly and weekly seasonality.
    Returns (model, forecast_df, metrics_dict).
    """
    # Train / test split (last 20% as holdout)
    split = int(len(ts) * 0.80)
    train = ts.iloc[:split]
    test  = ts.iloc[split:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq == "W"),
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.add_country_holidays(country_name="UK")
    model.fit(train)

    # Forecast on future dates
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Evaluate on hold-out set
    holdout_preds = forecast[forecast["ds"].isin(test["ds"])]
    merged = test.merge(holdout_preds[["ds", "yhat"]], on="ds", how="inner")

    mape = (np.abs(merged["y"] - merged["yhat"]) / merged["y"]).mean() * 100
    mae  = np.abs(merged["y"] - merged["yhat"]).mean()
    rmse = np.sqrt(((merged["y"] - merged["yhat"])**2).mean())

    metrics = {
        "MAPE":  round(mape, 2),
        "MAE":   round(mae, 2),
        "RMSE":  round(rmse, 2),
    }

    return model, forecast, train, test, metrics


def category_forecast(transactions: pd.DataFrame,
                       freq: str = "W") -> dict:
    """Fit individual Prophet models per product category."""
    results = {}
    cats = transactions["Category"].unique()
    for cat in cats:
        sub = transactions[transactions["Category"] == cat]
        ts  = prepare_prophet_df(sub, freq=freq)
        if len(ts) < 20:
            continue
        try:
            model, forecast, train, test, metrics = fit_prophet(ts, periods=12, freq=freq)
            results[cat] = {"forecast": forecast, "metrics": metrics, "train": train, "test": test}
        except Exception as e:
            print(f"  [WARNING] {cat}: {e}")
    return results
