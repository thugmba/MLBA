#
# Culstering (Zara)
# Daven
# 2025/5/27
#

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
za = pd.read_csv('zara.csv')

# Feature selection (x)
x = za[['Sales Volume', 'price']]

# Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Apply K-means clustering
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
za['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
custom_labels = ['Affordable Basics', 'Best Sellers', 'Premium Items']
for cluster in range(k_optimal):
    cluster_data = za[za['Cluster'] == cluster]
    plt.scatter(cluster_data['Sales Volume'], cluster_data['price'], label=custom_labels[cluster])
plt.title('Daven ML')
plt.xlabel('Sales Volume')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()