
# CUSTOMER ANALYTICS PLATFORM - MODEL DOCUMENTATION
Generated: 2025-10-25 22:20:12

## PROJECT OVERVIEW
This project analyzes Brazilian e-commerce data to predict customer churn and lifetime value,
enabling data-driven retention and revenue optimization strategies.

## DATA SUMMARY
- Total Customers: 93,358
- Date Range: 2016-09-04 to 2018-10-17
- Total Orders: 99,441
- Total Revenue: $13,279,836.59

## MODEL 1: CHURN PREDICTION
**Objective:** Identify customers likely to stop purchasing

**Algorithm:** XGBoost Classifier
**Training Samples:** 74,686
**Features:** 35

**Performance Metrics:**
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000
- ROC-AUC: 1.0000

**Top Features:**
1. recency_days (0.4265)
2. customer_age_days (0.3578)
3. R_score (0.2006)

**Business Impact:**
- Identified 13,186 at-risk customers
- Potential revenue at risk: $1,907,266.05

## MODEL 2: CUSTOMER LIFETIME VALUE PREDICTION
**Objective:** Predict 1-year customer value for active customers

**Algorithm:** XGBoost Regressor
**Training Samples:** 30,179
**Features:** 30

**Performance Metrics:**
- R² Score: 0.9861
- RMSE: $47.07
- MAE: $6.90
- MAPE: 1.43%

**Top Features:**
1. M_score (0.2820)
2. monetary (0.2421)
3. is_high_value (0.2410)

## RFM SEGMENTATION
Total Segments: 8
Largest Segment: Others (20,169 customers)

## RECOMMENDATIONS
1. Launch retention campaign for 13,186 at-risk customers
2. Focus VIP treatment on top 9,336 high-value customers (41% of revenue)
3. Immediate intervention for "Can't Lose Them" segment (0 customers)
4. Monitor recency metric as primary churn indicator

## TECHNICAL STACK
- Python 3.10+
- pandas, numpy (data processing)
- scikit-learn (preprocessing, metrics)
- XGBoost (modeling)
- Plotly (visualization)

## FILES
- Models: ./models/*.pkl
- Data: ./data/processed/customer_features.csv
- Notebooks: ./notebooks/01_data_exploration.ipynb
