from generate_data import generate_skills_sets, generate_user_skills, generate_graph
from clustering import clustering
from recommender import link_prediction

import numpy as np
import warnings

from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score
from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt


def evaluate_clustering():
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

    X = []
    Y = []
    nb_cluster_found = []

    max_skills_sets_sizes = 30
    for i in range(3, max_skills_sets_sizes):
        print(i)
        X.append(i)
        skills_sets = generate_skills_sets(i, 5, 7)
        users_skills, clusters_ground_truth = generate_user_skills(skills_sets, 500, 1, 2)

        clustering_range = (3, max_skills_sets_sizes)
        clustering_model = clustering(users_skills, range(*clustering_range), False)

        nb_cluster_found.append(len(clustering_model.cluster_centers_))

        info_score = normalized_mutual_info_score(clusters_ground_truth, clustering_model.labels_)
        Y.append(info_score)

    plt.figure()
    plt.title("")
    plt.xlabel("Number of skill sets")
    plt.ylabel("Normalized mutual info score")
    X = np.array(X)
    Y = np.array(Y)
    plt.plot(X, Y)

    is_correct = np.array(X) == np.array(nb_cluster_found)
    correct_indices = np.where(is_correct)[0]
    incorrect_indices = np.where(np.logical_not(is_correct))[0]
    plt.scatter(X[correct_indices], Y[correct_indices], color="blue", label="Correct number of cluster found")
    plt.scatter(X[incorrect_indices], Y[incorrect_indices], color="red", label="Incorrect number of cluster found")
    plt.ylim(0.95, 1.05)
    plt.legend()
    plt.savefig("clustering_evaluation.png")
    plt.show()



def evaluate_link_prediciton():
    print("Todo")

    # G = generate_graph(clusters_ground_truth)

    # Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
    # print("Clustering")
    #
    # print("- Number of clusters found", len(clustering_model.cluster_centers_))
    # print("- Real number of clusters", len(skills_sets))
    #
    # distance_function = "euclidean"
    # users_distances_to_centers = cdist(
    #     users_skills, clustering_model.cluster_centers_, metric=distance_function)
    #
    # print("Link prediction")
    # link_prediction_model = link_prediction(G, users_distances_to_centers)


if __name__ == '__main__':
    np.random.seed(int(np.pi * 42))
    np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

    evaluate_clustering()
