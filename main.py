import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# loading data
online_reatil_data = r'C:/Users/user/Desktop/Customer-Lifetime-Value-Prediction/OnlineRetail.xlsx'

# clean data, delete nulls, dublicates, change data types
df = pd.read_excel(online_reatil_data)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
# Visualize TotalAmount distribution
sns.histplot(df["TotalAmount"], bins=30, kde=True)
plt.title("Distribution of Total Spend per Customer")
# plt.show()

# the most money spent curstomers' top 10 list
customer_spending = df.groupby("CustomerID")["TotalAmount"].sum().sort_values(ascending=False)


# compute rfm(recency, frequency, monetay) features
reference_date = df["InvoiceDate"].max()
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency (days since last purchase)
    "InvoiceNo": "count",  # Frequency (total purchases)
    "TotalAmount": "sum"  # Monetary Value (total spending)
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

for col in rfm.columns:
    sns.histplot(rfm[col], bins=30, kde=True)
    plt.title(f"{col} Distribution")
    # plt.show()

# Normalize RFM features
scaler = MinMaxScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# Convert back to DataFrame
rfm_normalized = pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"], index=rfm.index)

rfm_normalized.head()

# Try different cluster sizes (K)
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_normalized)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
# plt.show()

# implement optimal K, in our case it is 2.
optimal_k = 2 # Change this based on the elbow plot

# Apply K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_normalized)

# Display the first few rows with cluster labels
rfm.head()

# Pairplot to visualize clusters
sns.pairplot(rfm, hue="Cluster", palette="viridis")
plt.suptitle("RFM Clusters", y=1.02)
# plt.show()

# Calculate mean RFM values for each cluster
cluster_summary = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).round(2)


# Assign business labels
rfm["Segment"] = rfm["Cluster"].map({
    0: "VIP Customers",
    1: "At-Risk Customers",
})

rfm["Frequency"] = rfm["Frequency"].replace(0, 1)  # Replace 0 with 1
  # Replace 0 with 1

plt.figure(figsize=(8, 5))
sns.countplot(x="Segment", data=rfm, palette="viridis")
plt.title("Customer Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
# plt.show()

# Calculate additional features
rfm["AveragePurchaseValue"] = rfm["Monetary"] / rfm["Frequency"]
rfm["CustomerAge"] = (df.groupby("CustomerID")["InvoiceDate"].max() - df.groupby("CustomerID")["InvoiceDate"].min()).dt.days
rfm.head()
# Calculate total purchases per customer
purchase_frequency = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
purchase_frequency.columns = ["CustomerID", "TotalPurchases"]

# Calculate customer lifetime (in days)
customer_lifetime = (df.groupby("CustomerID")["InvoiceDate"].max() - df.groupby("CustomerID")["InvoiceDate"].min()).dt.days.reset_index()
customer_lifetime.columns = ["CustomerID", "CustomerLifetime"]

# Drop rows with missing CustomerID in purchase_frequency and customer_lifetime
purchase_frequency = purchase_frequency[purchase_frequency["CustomerID"].notna()]
customer_lifetime = customer_lifetime[customer_lifetime["CustomerID"].notna()]

# Merge with rfm
rfm = rfm.merge(purchase_frequency, on="CustomerID", how="inner")
rfm = rfm.merge(customer_lifetime, on="CustomerID", how="inner")

# Calculate Purchase Frequency (purchases per month)
rfm["PurchaseFrequency"] = rfm["TotalPurchases"] / (rfm["CustomerLifetime"] / 30)  # Convert days to months
# Display the updated RFM DataFrame

# Define churn threshold (e.g., 90 days)
churn_threshold = 90

# Calculate days since last purchase
# Calculate the last purchase date for each customer
last_purchase_dates = df.groupby("CustomerID")["InvoiceDate"].max()

# Calculate the reference date (latest purchase date across all customers)
reference_date = last_purchase_dates.max()

# Calculate days since last purchase for each customer
# Align last_purchase_dates with rfm's index
print("Reference Date:", reference_date)
print("Last Purchase Dates Sample:")
print(last_purchase_dates.head())

print("Before Assignment:")
print((reference_date - last_purchase_dates).dt.days.head())

# Assign after fixing potential issues
rfm = rfm.reset_index()
last_purchase_dates = last_purchase_dates.reset_index()

rfm["DaysSinceLastPurchase"] = (reference_date - last_purchase_dates["InvoiceDate"]).dt.days


print("After Assignment:")
print(rfm["DaysSinceLastPurchase"].head())

rfm["Churned"] = (rfm["DaysSinceLastPurchase"] > churn_threshold).astype(int)


# Drop unnecessary columns
if "Segment" in rfm.columns:
    rfm_clv = rfm.drop(columns=["Segment"])
else:
    print("Column 'Segment' does not exist in the DataFrame.")
    rfm_clv = rfm.copy()
rfm_clv.replace([np.inf, -np.inf], np.nan, inplace=True)

rfm_clv = rfm_clv.dropna()

# Define features (X) and target (y)
X = rfm_clv.drop(columns=["Monetary", "CustomerID"])
y = rfm_clv["Monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest MSE:", mse_rf)
print("Random Forest R2:", r2_rf)
# Define parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# # Perform grid search
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring="r2")
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_
# print("Best parameters:", grid_search.best_params_)

# # Evaluate the best model
# y_pred_best = best_model.predict(X_test)
# mse_best = mean_squared_error(y_test, y_pred_best)
# r2_best = r2_score(y_test, y_pred_best)
# print("Best Model MSE:", mse_best)
# print("Best Model R2:", r2_best)

rfm_clv["Monetary"] = np.log1p(rfm_clv["Monetary"])  # Log transform Monetary

rfm_clv["Recency_Frequency"] = rfm_clv["Recency"] * rfm_clv["Frequency"]

# # Get feature importances from the tuned model
# feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
# feature_importances.sort_values(ascending=False).plot(kind="bar")
# plt.title("Feature Importances")
# plt.show()

# # Select top N features
# top_features = feature_importances.nlargest(5).index
# X_top = X[top_features]

from xgboost import XGBRegressor

# Initialize and train the model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost MSE:", mse_xgb)
print("XGBoost R2:", r2_xgb)

# from sklearn.model_selection import GridSearchCV

# # Define parameter grid
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [3, 5, 7],
#     "learning_rate": [0.01, 0.1, 0.2]
# }

# # Perform grid search
# grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=5, scoring="r2")
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_xgb_model = grid_search.best_estimator_
# print("Best parameters:", grid_search.best_params_)

# # Evaluate the best model
# y_pred_best_xgb = best_xgb_model.predict(X_test)
# mse_best_xgb = mean_squared_error(y_test, y_pred_best_xgb)
# r2_best_xgb = r2_score(y_test, y_pred_best_xgb)
# print("Best XGBoost MSE:", mse_best_xgb)
# print("Best XGBoost R2:", r2_best_xgb)

# from sklearn.model_selection import cross_val_score

# # Perform cross-validation
# scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")
# print("Cross-validated R2 scores:", scores)
# print("Mean R2:", scores.mean())
