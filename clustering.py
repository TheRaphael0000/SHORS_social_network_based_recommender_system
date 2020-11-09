from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from matplotlib import pyplot as plt
import math
from scipy.spatial.distance import cdist



def clustering(users_skills, n_clusters_range, plot=False):
    X = users_skills

    plotX = []
    plotY = []

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
        plotY.append(d)

    if plot:
        plt.xlabel("Nb clusters")
        plt.ylabel("davies_bouldin_score")
        plt.plot(plotX, plotY)
        plt.savefig("find_n_clusters_plot.png")

    return best_model

def get_distance_to_center(users_skills, centers):
    users_distances_to_centers = []
    for user_skills in users_skills:
        user_skills = [user_skills]
        distances = cdist(user_skills, centers)
        users_distances_to_centers.append(distances[0])

    return users_distances_to_centers
