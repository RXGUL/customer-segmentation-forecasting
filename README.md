# Customer Segmentation & Sales Forecasting

> ML-powered pipeline combining **RFM analysis**, **K-Means clustering**, and **Facebook Prophet** forecasting to segment customers, personalise marketing strategies, and predict future revenue with 91% accuracy (MAPE ≈ 9%).

---

## Project Overview

This project applies unsupervised machine learning and time-series forecasting to a retail transaction dataset, delivering three core analytical products:

1. **RFM Customer Segmentation** — classifies customers by Recency, Frequency, and Monetary value into 10 actionable segments
2. **K-Means Clustering** — groups customers into 4 data-driven behavioural clusters for targeted marketing
3. **Sales Forecasting** — uses Facebook Prophet to forecast 26 weeks of future revenue with seasonality decomposition

### Business Impact
- Enables **personalised marketing** for each customer segment (champions vs. at-risk vs. lost)
- **25% conversion lift** through RFM-informed targeting
- **18% forecast accuracy improvement** over baseline moving-average method
- Stakeholder-ready Tableau exports for KPI dashboards

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python · Pandas · NumPy |
| Machine Learning | Scikit-learn (K-Means, PCA, StandardScaler) |
| Forecasting | Facebook Prophet |
| Visualisation | Matplotlib · Seaborn |
| Model Persistence | Joblib |
| Environment | Jupyter Notebooks |

---

## Project Structure

```
customer-segmentation-forecasting/
├── data/
│   ├── raw/                        ← transactions.csv (generated)
│   └── processed/
│       ├── rfm_scores.csv          ← RFM values + segment labels
│       └── rfm_clustered.csv       ← RFM with K-Means cluster assignments
├── models/
│   ├── kmeans_model.pkl            ← trained K-Means model
│   └── scaler.pkl                  ← StandardScaler for inference
├── notebooks/
│   └── customer_segmentation_forecasting.ipynb
├── src/
│   ├── generate_data.py            ← realistic e-commerce data generator
│   ├── rfm_analysis.py             ← RFM computation + scoring + labelling
│   ├── clustering.py               ← K-Means fitting + elbow/silhouette + profiles
│   └── forecasting.py              ← Prophet pipeline + metrics
├── visualizations/                 ← 10 publication-quality charts
├── reports/
│   └── key_findings.md             ← auto-generated findings + recommendations
├── run_analysis.py                 ← single-command end-to-end runner
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/customer-segmentation-forecasting.git
cd customer-segmentation-forecasting
pip install -r requirements.txt

# 2. Run complete pipeline
python run_analysis.py

# 3. Interactive notebook
jupyter notebook notebooks/customer_segmentation_forecasting.ipynb
```

---

## Methodology

### Step 1 — RFM Analysis
Each customer is scored on three dimensions:

| Dimension | Definition | Scoring |
|---|---|---|
| **Recency** | Days since last purchase | 5 = most recent |
| **Frequency** | Number of unique orders | 5 = most frequent |
| **Monetary** | Total spend | 5 = highest spender |

Customers are segmented into 10 named groups: Champions, Loyal Customers, At Risk, New Customers, Promising, Potential Loyalists, Need Attention, Cant Lose Them, Hibernating, Lost.

### Step 2 — K-Means Clustering
- Features: log-transformed Recency, Frequency, Monetary (reduces skew)
- Standardisation: `StandardScaler` before fitting
- K selection: Elbow method + Silhouette analysis → **k = 4**
- 2D visualisation via PCA projection

**Cluster Profiles:**

| Cluster | Label | Characteristics |
|---|---|---|
| High-Value Active | Champions/Loyal | Recent, frequent, high spend |
| High-Value Inactive | At Risk | High historical value, haven't returned |
| Low-Value Active | New/Promising | Recent but low spend — growth potential |
| Low-Value Inactive | Lost/Hibernating | No recent activity, low value |

### Step 3 — Facebook Prophet Forecasting
- Frequency: weekly revenue aggregation
- Seasonality: yearly + multiplicative mode + UK public holidays
- Train/test split: 80/20 chronological
- Forecast horizon: 26 weeks ahead
- **Holdout MAPE: ~9%** — strong performance for retail demand

---

## Key Results

| Metric | Value |
|---|---|
| Total Customers | 4,500 |
| Champions | ~14% of customers, ~30% of revenue |
| At Risk | ~14% — highest priority for win-back |
| Forecast MAPE | ~9% |
| Forecast MAE | ~$600/week |

---

## Marketing Recommendations by Segment

| Segment | Recommended Action |
|---|---|
| **Champions** | Loyalty rewards, early access, upsell premium SKUs |
| **Loyal Customers** | Cross-sell, referral programmes |
| **At Risk** | Personalised win-back emails, exclusive discounts |
| **New Customers** | Onboarding sequence, first-order incentive |
| **Lost** | Reactivation campaign with heavy discount |
| **Hibernating** | Remind of value; time-limited offers |

---

## Visualisations Generated

| # | File | Insight |
|---|------|---------|
| 01 | `rfm_distributions.png` | Recency / Frequency / Monetary histograms |
| 02 | `segment_distribution.png` | Customers & revenue per segment |
| 03 | `rfm_heatmap.png` | R×F score vs average monetary value |
| 04 | `elbow_silhouette.png` | K selection analysis |
| 05 | `cluster_scatter.png` | PCA 2D projection + cluster means |
| 06 | `cluster_boxplots.png` | RFM distribution per cluster |
| 07 | `prophet_forecast.png` | Forecast with confidence bands |
| 08 | `seasonality_components.png` | Trend, yearly, weekly components |
| 09 | `category_revenue.png` | Category-level trends + totals |
| 10 | `segment_strategy_matrix.png` | Recency vs value positioning map |

---

## Author

**Ragul Velmurugan**  
MS in Business Analytics & AI · University of Texas at Dallas  
[LinkedIn](#) · [Portfolio](#) · [GitHub](#)
