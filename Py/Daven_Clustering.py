#
# Clustering iris dataset using KMeans
# Daven
# 2025/4/30
#

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

# 1. Input
iris = load_iris()
X = iris.data  # We use all four features

# 2. Process
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# 3. Output
plt.scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis', s=50)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Daven: KMeans Clustering on Iris Dataset')
plt.grid(True)
plt.show()
