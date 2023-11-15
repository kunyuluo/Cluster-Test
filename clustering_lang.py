import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('3.01.+Country+clusters.csv')

data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English': 0, 'French': 1, 'German': 2})
# print(data_mapped)

x = data_mapped.iloc[:, 3:4]
# print(x)

kmeans = KMeans(3)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)

data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
# print(data_with_clusters)

# plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
# plt.xlim(-180, 180)
# plt.ylim(-90, 90)
# plt.show()

# check the wcss (within-cluster sum of squares)
print(kmeans.inertia_)
