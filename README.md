# Customer Lifetime Value (CLV) Prediction Using Machine Learning

**Author:** Zulfizar Abdumurodova  
**Date:** October 25, 2023  

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Appendix](#appendix)

---

## Introduction

This project aims to predict **Customer Lifetime Value (CLV)** using transaction data from an online retail store. By analyzing **Recency, Frequency, and Monetary (RFM)** features, we built an **XGBoost model** to predict CLV and segment customers into **high**, **medium**, and **low-value** groups. The goal is to help businesses identify high-value customers and optimize marketing strategies.

### Problem Statement
Businesses often struggle to identify which customers are most valuable. By predicting CLV, we can segment customers and tailor marketing efforts to improve customer retention and revenue.

### Key Questions
1. What are the key drivers of customer lifetime value?
2. How can we segment customers based on their predicted CLV?

---

## Data Overview

### Dataset Description
The dataset contains **541,909 rows** and **8 columns**, including:
- `CustomerID`
- `InvoiceDate`
- `Quantity`
- `UnitPrice`

### Data Cleaning
- Removed missing values and duplicates.
- Handled negative quantities and unit prices.
- Calculated `TotalAmount` as `Quantity * UnitPrice`.

### Exploratory Data Analysis (EDA)
- The distribution of `TotalAmount` is highly skewed, with a few customers contributing significantly to revenue.
- Most customers have a low frequency of purchases.

### RFM Analysis
- **Recency:** Days since the last purchase.
- **Frequency:** Total number of purchases.
- **Monetary:** Total spending.

---

## Methodology

### Feature Engineering
- Created additional features:
  - `AveragePurchaseValue`: `Monetary / Frequency`
  - `CustomerAge`: Days since the first purchase.

### Model Selection
- **XGBoost** was chosen because it can handle non-linear relationships and is robust to outliers.

### Model Training and Evaluation
- Split the data into **training (80%)** and **testing (20%)** sets.
- Evaluated the model using **Mean Squared Error (MSE)** and **R-squared (R²)**.

### Cross-Validation
- Performed **5-fold cross-validation** to ensure the model generalizes well to unseen data.

### Hyperparameter Tuning
- Used **Grid Search** to find the optimal hyperparameters for the XGBoost model.

---

## Results

### Model Performance
- **Best XGBoost Model:**
  - **MSE:** 130,743,874.73
  - **R²:** 0.4948

### Visualizations
1. **Actual vs. Predicted CLV:**
   ![Actual vs. Predicted CLV](actual_vs_predicted.png)
   - The model’s predictions align well with the actual values.

2. **Feature Importances:**
   ![Feature Importances](feature_importances.png)
   - `Frequency` and `Monetary` are the most important features.

3. **Customer Segments:**
   ![Customer Segments](customer_segments.png)
   - Customers were segmented into **low**, **medium**, and **high CLV** groups.

### Key Insights
- The model identified **Frequency** and **Monetary** as the most important features.
- Customers were successfully segmented into **low**, **medium**, and **high CLV** groups.

---

## Conclusion

### Summary of Findings
The **XGBoost model** successfully predicted customer lifetime value, explaining **49.48%** of the variance in the data. Businesses can use this model to identify high-value customers and tailor marketing strategies to improve customer retention.

### Business Implications
- **High-value customers:** Focus on retention and loyalty programs.
- **Medium-value customers:** Encourage repeat purchases.
- **Low-value customers:** Target with promotions to increase engagement.

### Limitations
- The model’s performance could be improved with more data or additional features.

### Future Work
- Incorporate external data sources (e.g., demographics, website behavior).
- Experiment with other machine learning models (e.g., Gradient Boosting, Neural Networks).

---

## Appendix

### Code
The full code for this project can be found in the [Jupyter Notebook](clv_prediction.ipynb).

### Data
The dataset used in this project is available in the `data` folder.

### Saved Model
The best XGBoost model is saved as `clv_prediction_xgboost_model.pkl`.

---

**Thank you for reading!**  
For questions or feedback, please contact [Zulfizar Abdumurodova](mailto:arziqulovnaz@gamil.com).