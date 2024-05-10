import pandas as pd
import numpy as np

data = {
'transaction_id': [1,2,3,4,5,6],
'product': ['Milk', 'Egg', 'Apples', 'Curd', 'Bread', 'Bananas'], 
  'price': [23, 6, 40, 20, 50, 40],
'category': ['Dairy','Farm','Grocery','Dairy','Bakery','Grocery']
}
df = pd.DataFrame(data)
bins=[0, 20, 35, np.inf] 
labels=['Low', 'Medium', 'High']

df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)

print("\nDataFrame with discretized continuous variable (price):")
print(df)
