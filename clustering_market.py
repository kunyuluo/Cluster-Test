import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing
from WCSS import check_wcss

# Load data
data = pd.read_csv('market_data.csv')

# Preview data
# plt.scatter(data['Satisfaction'], data['Loyalty'])
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()

# Select feature
x = data.iloc[:, 0:2]
x_scaled = preprocessing.scale(x)
# print(x_scaled)

kmeans = KMeans(4)
kmeans.fit(x_scaled)
clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x_scaled)
# print(clusters)

# Clustering results
plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

# check_wcss(x_scaled)
