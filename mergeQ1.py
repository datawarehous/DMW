import pandas as pd
data1 = {
 'ID': [1, 2, 3],
 'Name': ['John', 'Alice', 'Bob'],
 'Score1': [80, 75, 90]
}

data2 = {
  'ID': [1, 2, 3], 
  'Score2': [85, 70, 95]
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

merged_df = pd.merge(df1, df2, on='ID', how='inner') 
merged_df.fillna(0, inplace=True)

merged_df['Total_Score'] = merged_df['Score1'] + merged_df['Score2'] 

print("Merged Dataset with Total Score:")
print(merged_df)
