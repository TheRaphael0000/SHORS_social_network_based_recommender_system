import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

from generate_data import get_named_synthetic_dataset
from clustering import clustering, fzclustering
from recommender import link_prediction
from visualization import visualization
from evaluation import print_evaluate


def use_case_kmeans(G, users_skills, clusters_ground_truth):
    clustering_range = (2, 10)
    distance_function = "euclidean"

    print("Clustering")
    print("Using KMeans")
    clustering_model = clustering(users_skills, range(*clustering_range), True)
    print("- Number of clusters found", len(clustering_model.cluster_centers_))
    print("- Real number of clusters", len(np.unique(clusters_ground_truth)))

    users_distances_to_centers = cdist(
        users_skills, clustering_model.cluster_centers_, metric=distance_function)

    print("Link prediction")
    model, y_train, predicted_train, y_test, predicted_test = link_prediction(
        G, users_distances_to_centers)

    print("Evaluation")
    print("- Train")
    print_evaluate(y_train, predicted_train)
    print("- Test")
    print_evaluate(y_test, predicted_test)

    print("Visualization")
    visualization(model, G, users_distances_to_centers,
                  clustering_model.labels_)


if __name__ == '__main__':
    np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
    np.random.seed(int(np.pi * 42))  # Seed for random number generation

    # Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
    G, users_skills, clusters_ground_truth = get_named_synthetic_dataset()
    use_case_kmeans(G, users_skills, clusters_ground_truth)
