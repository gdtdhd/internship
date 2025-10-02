import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create sample data
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Groceries': [200, 150, 400, 300, 100, 350, 250, 450, 120, 500],
    'Clothing': [80, 20, 60, 70, 10, 50, 40, 90, 30, 95],
    'Electronics': [500, 200, 100, 300, 150, 400, 250, 600, 100, 700],
    'HomeDecor': [60, 30, 90, 40, 20, 80, 60, 100, 25, 110]
}
df = pd.DataFrame(data)

# Preprocessing
features = df.drop('CustomerID', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_features)

df['PCA1'] = reduced[:, 0]
df['PCA2'] = reduced[:, 1]

sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments')
plt.show()

print(df[['CustomerID', 'Cluster']])