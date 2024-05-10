!pip install mlxtend


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules 
import pandas as pd
data = {
  'Transaction': [1, 2, 3, 4, 5], 
  'Items': [['Bread', 'Milk'],
              ['Bread', 'Diapers', 'Beer', 'Eggs'],
              ['Milk', 'Diapers', 'Beer', 'Cola'],
              ['Bread', 'Milk', 'Diapers', 'Beer'],
              ['Bread', 'Milk', 'Diapers', 'Cola']]
}
df = pd.DataFrame(data)
transactions = df['Items'].tolist() 
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions) 
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
association_rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
print("Frequent Itemsets:")
print(frequent_itemsets)
