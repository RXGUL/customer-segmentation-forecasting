# Customer Segmentation & Sales Forecasting – Key Findings

## Dataset Overview
- **Transactions**: 55,000
- **Unique Customers**: 4,500
- **Date Range**: 2022-01-01 → 2024-12-30
- **Total Revenue**: $1,075,078.56

## RFM Summary Statistics
| Metric    | Mean    | Median  | Std     |
|-----------|---------|---------|---------|
| Recency   | 78 days | 45 days | 92 |
| Frequency | 12.0 orders | 12.0 | 3.5 |
| Monetary  | $239 | $225 | $111 |

## Customer Segments
| Count               |   count |
|:--------------------|--------:|
| Loyal Customers     |    1252 |
| At Risk             |     640 |
| Champions           |     611 |
| New Customers       |     544 |
| Hibernating         |     473 |
| Lost                |     380 |
| Need Attention      |     207 |
| Promising           |     205 |
| Potential Loyalists |     110 |
| Cant Lose Them      |      78 |

## K-Means Cluster Profiles (k=4)
|   Cluster |   Recency |   Frequency |   Monetary | Cluster_Label       |
|----------:|----------:|------------:|-----------:|:--------------------|
|         3 |     30.83 |       15.76 |     349.44 | High-Value Active   |
|         2 |    120.65 |       11.76 |     241.72 | High-Value Inactive |
|         0 |     15.27 |       10.95 |     189.26 | Low-Value Active    |
|         1 |    153.31 |        7.44 |     113.09 | Low-Value Inactive  |

## Forecasting Performance (Facebook Prophet)
- **MAPE**: 8.94%
- **MAE**: $606
- **RMSE**: $725

## Marketing Recommendations
| Segment | Strategy |
|---------|----------|
| Champions | Reward loyalty; upsell premium products |
| Loyal Customers | Cross-sell; ask for referrals |
| At Risk | Win-back campaigns; personalised offers |
| New Customers | Onboarding journey; first-purchase incentives |
| Lost | Reactivation emails; heavy discount offers |
| Hibernating | Targeted promotions; remind of value proposition |
