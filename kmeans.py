import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

data = {
'customerid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'annual_income': [15000, 30000, 20000, 35000, 40000, 60000, 8000, 70000, 55000, 45000], 
  'spending_score': [39, 81, 6, 77, 40, 20, 15, 13, 62, 25]
}

df = pd.DataFrame(data)

X = df[['annual_income', 'spending_score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []

for k in range(1, 11):
kmeans = KMeans(n_clusters=k, random_state=42) 
kmeans.fit(X_scaled) 
inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o') 
plt.title('Elbow Method')
plt.xlabel('Number of Clusters') 
plt.ylabel('Inertia')
plt.show()

optimal_k = 5 
kmeans = KMeans(n_clusters=optimal_k, random_state=42) 
df['cluster'] = kmeans.fit_predict(X_scaled)

plt.scatter(df['annual_income'], df['spending_score'], c=df['cluster'], cmap='viridis', s=50, alpha=0.5, label='Data Points')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=300, c='red', label='Centroids')
plt.title('Clusters of Customers') 
plt.xlabel('Annual Income') 
plt.ylabel('Spending Score') 
plt.legend()
plt.grid(True) 
plt.show()

print(df)
