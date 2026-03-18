# Customer Segmentation & Forecasting

## Overview
This project looks at customer data to understand how different types of users behave and how their activity changes over time. The idea was to move beyond just looking at raw numbers and instead identify patterns that could actually help in decision-making.

---

## What this project is about
Businesses often have a lot of customer data but struggle to use it effectively. In this project, I focused on:
- Grouping customers based on their behaviour  
- Understanding how different segments contribute to overall activity  
- Observing patterns that could be useful for planning or targeting  

---

## How the analysis was done

### 1. Data preparation
- Cleaned the dataset and handled missing values  
- Made sure the data was consistent and usable  

### 2. Feature creation
- Built additional fields to better capture customer behaviour  
- Created simple indicators for activity, value, and engagement  

### 3. Segmentation
- Grouped customers based on similarity in behaviour  
- Looked at how different segments differ from each other  

### 4. Trend analysis
- Checked how customer activity changes over time  
- Identified patterns that could be useful for forecasting  

---

## Key takeaways

- A small group of customers tends to contribute a large portion of activity  
- Some segments show consistent behaviour, while others are more irregular  
- Patterns in customer activity can give a rough idea of future trends  
- Segmentation makes it easier to understand and compare different user groups  

---

---

## Project Structure

```
├── data/
├── notebooks/
├── src/
├── outputs/
├── reports/
├── visualizations/
├── README.md
├── run_analysis.py
```

---

## Quick Start

Clone the repository:
```bash
git clone https://github.com/RXGUL/customer-segmentation-forecasting.git
cd customer-segmentation-forecasting

---

## How the approach was built

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

**What each customer group looks like:**

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

## What stood out from the analysis

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

## Visual insights

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
