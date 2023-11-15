import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('3.01.+Country+clusters.csv')

# plt.scatter(data['Longitude'], data['Latitude'])
# plt.xlim(-180, 180)
# plt.ylim(-90, 90)
# plt.show()

x = data.iloc[:, 1:3]
# print(x)

kmeans = KMeans(3)
# kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
# print(identified_clusters)

print(kmeans.inertia_)

wcss = []

for i in range(1, 7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

print(wcss)

# Elbow methods
plt.plot(range(1, 7), wcss)
plt.show()
