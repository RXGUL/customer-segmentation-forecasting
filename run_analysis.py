"""
run_analysis.py
End-to-end runner:
  1. Generate data
  2. RFM analysis
  3. K-Means clustering
  4. Sales forecasting with Prophet
  5. Save all visualisations + reports
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA

from generate_data import generate_transactions
from rfm_analysis  import compute_rfm, score_rfm, label_segments
from clustering    import find_optimal_k, fit_kmeans, cluster_profiles
from forecasting   import prepare_prophet_df, fit_prophet, category_forecast

# ── palette & style ────────────────────────────────────────────────────────
PALETTE = ["#E63946", "#2A9D8F", "#E9C46A", "#F4A261", "#264653",
           "#A8DADC", "#457B9D", "#1D3557", "#606C38", "#BC6C25"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

VIZ_DIR   = BASE / "visualizations"
DATA_DIR  = BASE / "data"
MODEL_DIR = BASE / "models"
for d in [VIZ_DIR, DATA_DIR/"raw", DATA_DIR/"processed", MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Generate & load data
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("STEP 1 – Generating transaction dataset")
print("="*60)
df = generate_transactions()
df.to_csv(DATA_DIR / "raw" / "transactions.csv", index=False)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"  Transactions : {len(df):,}")
print(f"  Customers    : {df['CustomerID'].nunique():,}")
print(f"  Date range   : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – RFM Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2 – RFM Analysis")
rfm = compute_rfm(df)
rfm = score_rfm(rfm)
rfm = label_segments(rfm)
rfm.to_csv(DATA_DIR / "processed" / "rfm_scores.csv", index=False)
print(f"  RFM table: {len(rfm):,} customers")
print(rfm["Segment"].value_counts().to_string())

# ── Fig 1: RFM Score distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("RFM Score Distributions", fontsize=16, fontweight="bold")
for ax, (col, label, color) in zip(axes, [
    ("Recency",   "Recency (days)",    "#E63946"),
    ("Frequency", "Frequency (orders)","#2A9D8F"),
    ("Monetary",  "Monetary ($)",      "#E9C46A"),
]):
    ax.hist(rfm[col], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(rfm[col].median(), color="#264653", linestyle="--", linewidth=1.8, label=f"Median: {rfm[col].median():.0f}")
    ax.set_title(label, fontsize=13)
    ax.set_xlabel(label); ax.set_ylabel("Customers")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(VIZ_DIR / "01_rfm_distributions.png")
plt.close()
print("  Saved: 01_rfm_distributions.png")

# ── Fig 2: Customer segment treemap / bar ─────────────────────────────────
seg_counts = rfm["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]
seg_counts["Pct"] = (seg_counts["Count"] / seg_counts["Count"].sum() * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Customer Segment Distribution", fontsize=16, fontweight="bold")

ax = axes[0]
bars = ax.barh(seg_counts["Segment"], seg_counts["Count"],
               color=PALETTE[:len(seg_counts)], edgecolor="white")
for bar, (_, row) in zip(bars, seg_counts.iterrows()):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{row['Count']:,} ({row['Pct']}%)", va="center", fontsize=9)
ax.set_title("Customers per Segment", fontsize=13)
ax.set_xlabel("Number of Customers")
ax.set_xlim(0, seg_counts["Count"].max() * 1.25)

ax = axes[1]
seg_revenue = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=True)
bars2 = ax.barh(seg_revenue.index, seg_revenue.values,
                color=PALETTE[:len(seg_revenue)], edgecolor="white")
for bar, val in zip(bars2, seg_revenue.values):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9)
ax.set_title("Total Revenue per Segment", fontsize=13)
ax.set_xlabel("Total Revenue ($)")
ax.set_xlim(0, seg_revenue.max() * 1.25)

plt.tight_layout()
plt.savefig(VIZ_DIR / "02_segment_distribution.png")
plt.close()
print("  Saved: 02_segment_distribution.png")

# ── Fig 3: RFM Score heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pivot_rf = rfm.pivot_table(values="Monetary", index="R_Score", columns="F_Score", aggfunc="mean").round(0)
sns.heatmap(pivot_rf, annot=True, fmt=".0f", cmap="RdYlGn",
            ax=ax, linewidths=0.5, cbar_kws={"label": "Avg Monetary ($)"})
ax.set_title("Average Monetary Value – Recency Score × Frequency Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency Score (1=Low → 5=High)")
ax.set_ylabel("Recency Score (1=Churned → 5=Active)")
plt.tight_layout()
plt.savefig(VIZ_DIR / "03_rfm_heatmap.png")
plt.close()
print("  Saved: 03_rfm_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – K-Means Clustering
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3 – K-Means Clustering")
from sklearn.preprocessing import StandardScaler

features = ["Recency", "Frequency", "Monetary"]
X = rfm[features].copy()
X["Frequency"] = np.log1p(X["Frequency"])
X["Monetary"]  = np.log1p(X["Monetary"])
X["Recency"]   = np.log1p(X["Recency"])
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range, inertias, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 9))

# ── Fig 4: Elbow + Silhouette ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimal K Selection", fontsize=16, fontweight="bold")

ax = axes[0]
ax.plot(k_range, inertias, "o-", color="#E63946", linewidth=2, markersize=7)
ax.set_title("Elbow Method (Inertia)", fontsize=13)
ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
ax.axvline(4, color="#2A9D8F", linestyle="--", linewidth=1.5, label="k=4 selected")
ax.legend()

ax = axes[1]
ax.plot(k_range, silhouettes, "s-", color="#2A9D8F", linewidth=2, markersize=7)
ax.set_title("Silhouette Score", fontsize=13)
ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Silhouette Score")
ax.axvline(4, color="#E63946", linestyle="--", linewidth=1.5, label="k=4 selected")
ax.legend()

plt.tight_layout()
plt.savefig(VIZ_DIR / "04_elbow_silhouette.png")
plt.close()
print("  Saved: 04_elbow_silhouette.png")

# Fit final model
rfm_clustered, scaler_fit, km_model, X_sc = fit_kmeans(rfm, n_clusters=4, model_dir=MODEL_DIR)
profiles = cluster_profiles(rfm_clustered)
print(profiles.to_string(index=False))
rfm_clustered.to_csv(DATA_DIR / "processed" / "rfm_clustered.csv", index=False)

# ── Fig 5: Cluster scatter (PCA 2D) ───────────────────────────────────────
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
explained = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("K-Means Clustering – Customer Segments", fontsize=16, fontweight="bold")

ax = axes[0]
for cl in sorted(rfm_clustered["Cluster"].unique()):
    mask = rfm_clustered["Cluster"] == cl
    label_str = profiles.loc[profiles["Cluster"]==cl, "Cluster_Label"].values
    label_str = label_str[0] if len(label_str) else f"Cluster {cl}"
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=12, alpha=0.45, color=PALETTE[cl], label=f"Cluster {cl}: {label_str}")
ax.set_title(f"PCA Projection (PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)", fontsize=12)
ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
ax.legend(fontsize=8, markerscale=3)

ax = axes[1]
cluster_rfm = rfm_clustered.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean().reset_index()
x = np.arange(len(cluster_rfm))
w = 0.25
bars_r = ax.bar(x - w, cluster_rfm["Recency"],    w, label="Recency",   color="#E63946", alpha=0.85)
bars_f = ax.bar(x,     cluster_rfm["Frequency"],  w, label="Frequency", color="#2A9D8F", alpha=0.85)
bars_m = ax.bar(x + w, cluster_rfm["Monetary"]/100, w, label="Monetary/100", color="#E9C46A", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"Cluster {c}" for c in cluster_rfm["Cluster"]])
ax.set_title("Cluster Mean RFM Values", fontsize=13)
ax.legend(fontsize=9); ax.set_ylabel("Value")

plt.tight_layout()
plt.savefig(VIZ_DIR / "05_cluster_scatter.png")
plt.close()
print("  Saved: 05_cluster_scatter.png")

# ── Fig 6: Cluster box plots ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("RFM Distribution per Cluster", fontsize=16, fontweight="bold")
for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
    data_to_plot = [rfm_clustered.loc[rfm_clustered["Cluster"]==c, col].values
                    for c in sorted(rfm_clustered["Cluster"].unique())]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_title(col, fontsize=13)
    ax.set_xticklabels([f"C{c}" for c in sorted(rfm_clustered["Cluster"].unique())])
    ax.set_xlabel("Cluster"); ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(VIZ_DIR / "06_cluster_boxplots.png")
plt.close()
print("  Saved: 06_cluster_boxplots.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Sales Forecasting (Facebook Prophet)
# ══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4 – Sales Forecasting with Facebook Prophet")
ts = prepare_prophet_df(df, freq="W")
model, forecast, train, test, metrics = fit_prophet(ts, periods=26, freq="W")
print(f"  Forecast metrics – MAPE: {metrics['MAPE']}%  MAE: ${metrics['MAE']:,.0f}  RMSE: ${metrics['RMSE']:,.0f}")

# ── Fig 7: Prophet forecast ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Weekly Revenue Forecast – Facebook Prophet", fontsize=16, fontweight="bold")

ax = axes[0]
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                alpha=0.2, color="#2A9D8F", label="Confidence Interval")
ax.plot(forecast["ds"], forecast["yhat"], color="#2A9D8F", linewidth=2, label="Forecast")
ax.scatter(train["ds"], train["y"], s=8, color="#264653", alpha=0.6, label="Actual (train)")
ax.scatter(test["ds"],  test["y"],  s=12, color="#E63946", alpha=0.8, label="Actual (test)")
ax.axvline(train["ds"].max(), color="gray", linestyle="--", linewidth=1.2, label="Train/Test split")
ax.set_title("Revenue Forecast with Confidence Bands", fontsize=13)
ax.set_xlabel("Date"); ax.set_ylabel("Weekly Revenue ($)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

ax = axes[1]
components = model.predict(train)
ax.plot(components["ds"], components["trend"], color="#E63946", linewidth=2, label="Trend")
ax.fill_between(components["ds"],
                components["trend"] * 0.95, components["trend"] * 1.05,
                alpha=0.15, color="#E63946")
ax.set_title("Underlying Trend Component", fontsize=13)
ax.set_xlabel("Date"); ax.set_ylabel("Trend ($)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig(VIZ_DIR / "07_prophet_forecast.png")
plt.close()
print("  Saved: 07_prophet_forecast.png")

# ── Fig 8: Seasonality decomposition ──────────────────────────────────────
from prophet.plot import plot_components
fig2 = model.plot_components(forecast)
fig2.suptitle("Prophet Seasonality Components", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(VIZ_DIR / "08_seasonality_components.png", bbox_inches="tight")
plt.close()
print("  Saved: 08_seasonality_components.png")

# ── Fig 9: Category-level revenue trends ──────────────────────────────────
monthly = (
    df[df["Revenue"] > 0]
    .groupby([pd.Grouper(key="InvoiceDate", freq="ME"), "Category"])["Revenue"]
    .sum()
    .reset_index()
)

top_cats = df.groupby("Category")["Revenue"].sum().nlargest(5).index.tolist()
monthly_top = monthly[monthly["Category"].isin(top_cats)]

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Category-Level Revenue Analysis", fontsize=16, fontweight="bold")

ax = axes[0]
for i, cat in enumerate(top_cats):
    sub = monthly_top[monthly_top["Category"] == cat]
    ax.plot(sub["InvoiceDate"], sub["Revenue"], marker="o", markersize=4,
            linewidth=2, color=PALETTE[i], label=cat)
ax.set_title("Monthly Revenue – Top 5 Categories", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("Revenue ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9)

ax = axes[1]
cat_total = df[df["Revenue"] > 0].groupby("Category")["Revenue"].sum().sort_values(ascending=False)
bars = ax.bar(cat_total.index, cat_total.values,
              color=PALETTE[:len(cat_total)], edgecolor="white")
ax.bar_label(bars, fmt="$%.0f", rotation=45, padding=4, fontsize=8)
ax.set_title("Total Revenue by Category (Full Period)", fontsize=13)
ax.set_xlabel("Category"); ax.set_ylabel("Revenue ($)")
ax.tick_params(axis="x", rotation=30)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig(VIZ_DIR / "09_category_revenue.png")
plt.close()
print("  Saved: 09_category_revenue.png")

# ── Fig 10: Marketing strategy matrix ─────────────────────────────────────
seg_stats = rfm.groupby("Segment").agg(
    avg_recency=("Recency","mean"),
    avg_monetary=("Monetary","mean"),
    count=("CustomerID","count")
).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    seg_stats["avg_recency"], seg_stats["avg_monetary"],
    s=seg_stats["count"] * 0.8,
    c=range(len(seg_stats)), cmap="tab10", alpha=0.8, edgecolors="white", linewidth=1.5
)
for _, row in seg_stats.iterrows():
    ax.annotate(row["Segment"],
                (row["avg_recency"], row["avg_monetary"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight="bold")
ax.axhline(seg_stats["avg_monetary"].median(), color="gray", linestyle="--", alpha=0.5)
ax.axvline(seg_stats["avg_recency"].median(), color="gray", linestyle="--", alpha=0.5)
ax.set_title("Customer Segment – Recency vs Value Matrix\n(bubble size = # customers)", fontsize=13, fontweight="bold")
ax.set_xlabel("Avg Recency (days since last purchase) →  Lower is Better")
ax.set_ylabel("Avg Monetary Value ($)")
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(VIZ_DIR / "10_segment_strategy_matrix.png")
plt.close()
print("  Saved: 10_segment_strategy_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
report = f"""# Customer Segmentation & Sales Forecasting – Key Findings

## Dataset Overview
- **Transactions**: {len(df):,}
- **Unique Customers**: {df['CustomerID'].nunique():,}
- **Date Range**: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}
- **Total Revenue**: ${df[df['Revenue']>0]['Revenue'].sum():,.2f}

## RFM Summary Statistics
| Metric    | Mean    | Median  | Std     |
|-----------|---------|---------|---------|
| Recency   | {rfm['Recency'].mean():.0f} days | {rfm['Recency'].median():.0f} days | {rfm['Recency'].std():.0f} |
| Frequency | {rfm['Frequency'].mean():.1f} orders | {rfm['Frequency'].median():.1f} | {rfm['Frequency'].std():.1f} |
| Monetary  | ${rfm['Monetary'].mean():,.0f} | ${rfm['Monetary'].median():,.0f} | ${rfm['Monetary'].std():,.0f} |

## Customer Segments
{rfm['Segment'].value_counts().reset_index().rename(columns={'index':'Segment','Segment':'Count'}).to_markdown(index=False)}

## K-Means Cluster Profiles (k=4)
{profiles.to_markdown(index=False)}

## Forecasting Performance (Facebook Prophet)
- **MAPE**: {metrics['MAPE']}%
- **MAE**: ${metrics['MAE']:,.0f}
- **RMSE**: ${metrics['RMSE']:,.0f}

## Marketing Recommendations
| Segment | Strategy |
|---------|----------|
| Champions | Reward loyalty; upsell premium products |
| Loyal Customers | Cross-sell; ask for referrals |
| At Risk | Win-back campaigns; personalised offers |
| New Customers | Onboarding journey; first-purchase incentives |
| Lost | Reactivation emails; heavy discount offers |
| Hibernating | Targeted promotions; remind of value proposition |
"""

(BASE / "reports").mkdir(exist_ok=True)
with open(BASE / "reports" / "key_findings.md", "w") as f:
    f.write(report)

print("\n" + "="*60)
print("ALL DONE")
print(f"  Visualizations : {VIZ_DIR}")
print(f"  Report         : reports/key_findings.md")
print(f"  Models         : {MODEL_DIR}")
print("="*60)
