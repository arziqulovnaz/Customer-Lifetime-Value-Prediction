import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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

print(rfm_normalized.head())