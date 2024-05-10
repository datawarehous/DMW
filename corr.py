import numpy as np
import pandas as pd
from sklearn.datasets import make_classification 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

np.random.seed(42)

X, _ = make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=2, random_state=42) 
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]) 
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', interpolation='none',aspect='auto') 
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90) 
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

pca = PCA() X_pca=pca.fit_transform(scaled_data)

explained_variance_ratio = pca.explained_variance_ratio_ 
print("Explained Variance Ratio :",explained_variance_ratio)

plt.figure(figsize=(8, 6)) 
plt.plot(np.cumsum(explained_variance_ratio))
plt.title('Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio Vs Number of principal Components')
plt.show()
