"""
main.py
-------
End-to-end pipeline:
  1. Generate synthetic e-commerce transactions
  2. Compute RFM scores
  3. K-Means clustering (with elbow selection)
  4. Facebook Prophet sales forecasting
  5. Export all figures + Tableau-ready CSVs

Usage
-----
    python main.py
    python main.py --skip-forecast     # skip Prophet (faster)
    python main.py --k 5               # force number of clusters
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import load_or_generate
from rfm_analysis   import compute_rfm, rfm_segment_summary, save_rfm
from clustering     import (prepare_features, elbow_method, fit_kmeans,
                             attach_clusters, cluster_profiles,
                             plot_elbow, plot_cluster_scatter_pca,
                             plot_rfm_boxplots, plot_segment_revenue,
                             save_clustered)
from forecasting    import (aggregate_daily_revenue, aggregate_monthly_revenue,
                             fit_and_forecast, evaluate_model,
                             plot_forecast, plot_forecast_components,
                             plot_category_revenue, plot_segment_forecast,
                             export_tableau_forecast)

TABLEAU_DIR   = os.path.join(os.path.dirname(__file__), "outputs", "tableau_ready")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run(skip_forecast: bool = False, k_override: int | None = None):
    os.makedirs(TABLEAU_DIR,   exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print_section("STEP 1: Data Generation")
    df = load_or_generate()
    print(f"\nTransaction sample:\n{df.head(3).to_string()}\n")

    # ── 2. RFM Analysis ────────────────────────────────────────────────────────
    print_section("STEP 2: RFM Analysis")
    rfm = compute_rfm(df)
    seg_summary = rfm_segment_summary(rfm)
    print(f"\n[RFM Segment Summary]\n{seg_summary.to_string()}")
    save_rfm(rfm)

    # ── 3. Clustering ──────────────────────────────────────────────────────────
    print_section("STEP 3: K-Means Clustering")
    X_scaled, scaler, rfm_numeric = prepare_features(rfm)

    # Elbow method
    elbow_df = elbow_method(X_scaled, k_range=range(2, 9))
    plot_elbow(elbow_df)

    # Optimal k: highest silhouette
    if k_override:
        best_k = k_override
    else:
        best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])
    print(f"\n[clustering] Selected k = {best_k}")

    km = fit_kmeans(X_scaled, k=best_k)
    rfm_clustered = attach_clusters(rfm, km.labels_)
    profile = cluster_profiles(rfm_clustered)

    # Visualisations
    print("\n[clustering] Generating cluster visualisations …")
    plot_cluster_scatter_pca(X_scaled, km.labels_, profile)
    plot_rfm_boxplots(rfm_clustered, profile)
    plot_segment_revenue(profile)

    # Exports
    save_clustered(rfm_clustered)
    rfm_clustered.to_csv(
        os.path.join(TABLEAU_DIR, "rfm_clustered.csv"), index=False
    )
    profile.to_csv(
        os.path.join(TABLEAU_DIR, "cluster_profiles.csv"), index=False
    )
    seg_summary.to_csv(
        os.path.join(TABLEAU_DIR, "rfm_segment_summary.csv"), index=False
    )

    # ── 4. Sales Forecasting ───────────────────────────────────────────────────
    if not skip_forecast:
        print_section("STEP 4: Sales Forecasting (Prophet)")
        try:
            weekly_df = aggregate_monthly_revenue(df)
            daily_df  = aggregate_daily_revenue(df)

            # Fit on weekly for cleaner signal
            forecast, future_only, model = fit_and_forecast(
                weekly_df, periods=26, freq="W"
            )

            # Visualisations
            print("[forecasting] Generating forecast visualisations …")
            plot_forecast(weekly_df, forecast, future_only)
            plot_forecast_components(model, forecast)
            plot_category_revenue(df)
            plot_segment_forecast(rfm_clustered, df)

            # Export
            export_tableau_forecast(forecast, weekly_df)
            df.to_csv(os.path.join(TABLEAU_DIR, "transactions.csv"), index=False)

            # Evaluation (quick — 1 fold)
            print("\n[forecasting] Running cross-validation …")
            metrics = evaluate_model(
                weekly_df, initial="78 weeks", period="13 weeks", horizon="13 weeks"
            )
            if not metrics.empty:
                metrics.to_csv(
                    os.path.join(PROCESSED_DIR, "cv_metrics.csv"), index=False
                )

        except Exception as e:
            print(f"[forecasting] ⚠ Prophet step failed: {e}")
            print("  Tip: pip install prophet  (requires pystan)")
    else:
        print_section("STEP 4: Forecasting SKIPPED")
        # Still generate category chart without Prophet
        plot_category_revenue(df)
        df.to_csv(os.path.join(TABLEAU_DIR, "transactions.csv"), index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_section("PIPELINE COMPLETE ✓")
    print("  Figures      →  outputs/figures/")
    print("  Tableau CSVs →  outputs/tableau_ready/")
    print("  Processed    →  data/processed/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Segmentation & Sales Forecasting Pipeline"
    )
    parser.add_argument("--skip-forecast", action="store_true",
                        help="Skip Prophet forecasting (faster)")
    parser.add_argument("--k", type=int, default=None,
                        help="Force number of clusters (default: auto from silhouette)")
    args = parser.parse_args()
    run(skip_forecast=args.skip_forecast, k_override=args.k)
