# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Step 1: Load and clean data
# Load data
df = pd.read_excel("OnlineRetail.xlsx")

# Clean data (drop missing values, duplicates, etc.)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Step 2: Calculate RFM features
# Calculate RFM features
reference_date = df["InvoiceDate"].max()
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
    "InvoiceNo": "count",  # Frequency
    "TotalAmount": "sum"  # Monetary
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

# Step 3: Add additional features
# Add additional features (e.g., AveragePurchaseValue, CustomerAge, etc.)
rfm["AveragePurchaseValue"] = rfm["Monetary"] / rfm["Frequency"]
rfm["CustomerAge"] = (df.groupby("CustomerID")["InvoiceDate"].max() - df.groupby("CustomerID")["InvoiceDate"].min()).dt.days

# Step 4: Handle missing/infinite values
# Replace infinite values with NaN and drop missing values
rfm.replace([np.inf, -np.inf], np.nan, inplace=True)
rfm.dropna(inplace=True)

# Step 5: Define features and target
# Define features (X) and target (y)
X = rfm.drop(columns=["Monetary"])
y = rfm["Monetary"]

# Step 6: Split the data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the XGBoost model
# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Step 8: Make predictions
# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Step 9: Evaluate the model
# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("XGBoost MSE:", mse)
print("XGBoost R2:", r2)

# Step 10: Perform cross-validation
# Perform 5-fold cross-validation
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")
print("Cross-validated R2 scores:", scores)
print("Mean R2:", scores.mean())

# Step 11: Hyperparameter tuning (optional)
# Define parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

# Initialize Grid Search
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="r2")

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_xgb_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate the best model
y_pred_best = best_xgb_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
print("Best XGBoost MSE:", mse_best)
print("Best XGBoost R2:", r2_best)

# Step 12: Visualize results
# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual CLV")
plt.ylabel("Predicted CLV")
plt.title("Actual vs. Predicted CLV")
plt.show()

# Plot feature importances
feature_importances = pd.Series(best_xgb_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind="bar")
plt.title("Feature Importances")
plt.show()

# Step 13: Save the model
# Save the best model
joblib.dump(best_xgb_model, "clv_prediction_xgboost_model.pkl")

# Step 14: Segment customers based on predicted CLV
# Add predicted CLV to the original DataFrame
rfm["Predicted_CLV"] = best_xgb_model.predict(X)

# Segment customers into low, medium, and high CLV
rfm["CLV_Segment"] = pd.cut(rfm["Predicted_CLV"], bins=[0, 100, 500, np.inf], labels=["Low", "Medium", "High"])

# Visualize segments
sns.countplot(x="CLV_Segment", data=rfm, palette="viridis")
plt.title("Customer Segments Based on Predicted CLV")
plt.show()