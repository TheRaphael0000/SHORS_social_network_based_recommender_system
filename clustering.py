from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from matplotlib import pyplot as plt
import math


def clustering(users_skills, n_clusters_range, plot=False):
    X = users_skills

    plotX = []
    plotY_d = []
    plotY_s = []

    # Record the best clustering
    best_d = math.inf
    best_model = None

    # Find the best number of clusters
    for n in n_clusters_range:
        kmeans = KMeans(n_clusters=n, random_state=42,
                        n_init=3, max_iter=1000).fit(X)
        d = davies_bouldin_score(X, kmeans.labels_)
        s = silhouette_score(X, kmeans.labels_)

        if d < best_d:
            best_d = d
            best_model = kmeans

        plotX.append(n)
        plotY_d.append(d)
        plotY_s.append(s)

    if plot:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Nb clusters")
        ax2 = ax1.twinx()
        color1 = "tab:blue"
        color2 = "tab:orange"
        ax1.plot(plotX, plotY_d, label="davies_bouldin_score", color=color1)
        ax1.set_ylabel("davies_bouldin_score", color=color1)
        ax2.plot(plotX, plotY_s, label="silhouette_score", color=color2)
        ax2.set_ylabel("silhouette_score", color=color2)
        fig.legend()
        fig.tight_layout()
        fig.savefig("find_n_clusters_plot.png")

    return best_model
