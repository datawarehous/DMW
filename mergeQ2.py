import pandas as pd

d1 = pd.read_csv("C:\\Users\\soham\\Desktop\\csv\\D1.csv")

d2 = pd.read_csv("C:\\Users\\soham\\Desktop\\csv\\D2.csv")

merged_df = pd.merge(d1, d2, on='ID', how='inner')

print("Merged Dataset:")
print(merged_df)

merged_df.to_csv("C:\\Users\\soham\\Desktop\\csv\\newD3.csv", index=False) 
print("Merged dataset saved to `Mergedset2.csv`")
