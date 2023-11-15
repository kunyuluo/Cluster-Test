from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def check_wcss(x):
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
