import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

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
rfm.head()

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

print(cluster_summary)

# Assign business labels
rfm["Segment"] = rfm["Cluster"].map({
    0: "VIP Customers",
    1: "At-Risk Customers",
    2: "Regular Customers",
    3: "Big Spenders"
})

# Display the first few rows with segments
rfm.head()

plt.figure(figsize=(8, 5))
sns.countplot(x="Segment", data=rfm, palette="viridis")
plt.title("Customer Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.show()

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

# Merge with RFM data
rfm = rfm.merge(purchase_frequency, on="CustomerID")
rfm = rfm.merge(customer_lifetime, on="CustomerID")

# Calculate Purchase Frequency (purchases per month)
rfm["PurchaseFrequency"] = rfm["TotalPurchases"] / (rfm["CustomerLifetime"] / 30)  # Convert days to months

# Display the updated RFM DataFrame
rfm.head()

# Define churn threshold (e.g., 90 days)
churn_threshold = 90

# Calculate days since last purchase
rfm["DaysSinceLastPurchase"] = (df.groupby("CustomerID")["InvoiceDate"].max().max() - df.groupby("CustomerID")["InvoiceDate"].max()).dt.days

# Define churn (1 = churned, 0 = not churned)
rfm["Churned"] = (rfm["DaysSinceLastPurchase"] > churn_threshold).astype(int)

# Display the updated RFM DataFrame
rfm.head()

print(rfm.head())