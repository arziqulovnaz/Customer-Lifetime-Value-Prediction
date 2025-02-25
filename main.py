import pandas as pd

online_reatil_data = r'C:/Users/user/Desktop/Customer-Lifetime-Value-Prediction/.venv/OnlineRetail.xlsx'

df = pd.read_excel(online_reatil_data)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Quantity'] = df['Quantity'].astype(float)
print(df.info())