import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/python/TASK8'):
    for filename in filenames:
        print(os.path.join(dirname,filename))
        
        
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv('/python/TASK8/Mall_Customers.csv')

data.head()


data.info()
data.shape

data.describe()
print(data.isnull().sum())

data.drop('CustomerID',axis=1)

label= LabelEncoder()

data['Gender'] = label.fit_transform(data['Gender'])


iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(data)
data = data[outliers == 1]  # Keep non-outliers
print("Number of outliers removed:", sum(outliers == -1))



scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


# Plot Elbow and Silhouette
plt.figure(figsize=(14, 7))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for K-Means')
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)
kmeans_silhouette = silhouette_score(scaled_data, kmeans.labels_)
print("K-Means Silhouette Score:", kmeans_silhouette)


dbscan = DBSCAN(eps=0.39, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)
if len(np.unique(dbscan.labels_[dbscan.labels_ != -1])) > 1:
    dbscan_silhouette = silhouette_score(scaled_data[dbscan.labels_ != -1], 
                                        dbscan.labels_[dbscan.labels_ != -1])
    print("DBSCAN Silhouette Score:", dbscan_silhouette)
else:
    print("DBSCAN produced too few clusters for silhouette score.")
    
    
    
    
bic = []
N = range(2, 16)
for n in N:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(scaled_data)
    bic.append(gmm.bic(scaled_data))

# Plot BIC Curve
plt.figure(figsize=(6, 4))
plt.plot(N, bic, 'ro-')
plt.xlabel('Number of Components (n)')
plt.ylabel('BIC')
plt.title('BIC for GMM')
plt.show()



gmm = GaussianMixture(n_components=14, random_state=42)
gmm_labels = gmm.fit_predict(scaled_data)  
data['GMM_Cluster'] = gmm_labels
gmm_silhouette = silhouette_score(scaled_data, gmm_labels)  
print("GMM Silhouette Score:", gmm_silhouette)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
data['PCA1'] = pca_data[:, 0]
data['PCA2'] = pca_data[:, 1]
print("\nExplained Variance Ratio by PCA:", pca.explained_variance_ratio_)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='Set1', data=data)
plt.title('K-Means Clusters (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 3, 2)
sns.scatterplot(x='PCA1', y='PCA2', hue='DBSCAN_Cluster', palette='Set2', data=data)
plt.title('DBSCAN Clusters (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 3, 3)
sns.scatterplot(x='PCA1', y='PCA2', hue='GMM_Cluster', palette='Set3', data=data)
plt.title('GMM Clusters (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.show()



algorithms = ['K-Means', 'DBSCAN', 'GMM']
silhouette_scores = [kmeans_silhouette, dbscan_silhouette, gmm_silhouette]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, silhouette_scores, color=['blue', 'orange', 'green'])
plt.xlabel('Algorithms')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Comparison')
for i, score in enumerate(silhouette_scores):
    plt.text(i, score, f'{score:.3f}', ha='center', va='bottom' if score > 0 else 'top')
plt.show()