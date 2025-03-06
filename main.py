import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# loading data
online_reatil_data = r'C:/Users/user/Desktop/Customer-Lifetime-Value-Prediction/OnlineRetail.xlsx'

# clean data, delete nulls, dublicates, change data types
df = pd.read_excel(online_reatil_data)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Quantity'] = df['Quantity'].astype(float)

# Visualize TotalAmount distribution
sns.histplot(df["UnitPrice"], bins=30, kde=True)
plt.title("Distribution of Total Spend per Customer")
# plt.show()

# the most money spent curstomers' top 10 list
customer_spending = df.groupby("CustomerID")["UnitPrice"].sum().sort_values(ascending=False)


# compute rfm(recency, frequency, monetay) features
reference_date = df["InvoiceDate"].max()
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency (days since last purchase)
    "InvoiceNo": "count",  # Frequency (total purchases)
    "UnitPrice": "sum"  # Monetary Value (total spending)
})

rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm.head()

# Recency Distribution
sns.histplot(rfm["Recency"], bins=30, kde=True)
plt.title("Recency Distribution")
plt.show()

# Frequency Distribution
sns.histplot(rfm["Frequency"], bins=30, kde=True)
plt.title("Frequency Distribution")
plt.show()

# Monetary Value Distribution
sns.histplot(rfm["Monetary"], bins=30, kde=True)
plt.title("Monetary Value Distribution")
plt.show()
