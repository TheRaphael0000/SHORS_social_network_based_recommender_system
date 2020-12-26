from generate_data import generate_skills_sets, generate_user_skills, generate_graph
from clustering import clustering
from recommender import link_prediction

import numpy as np
import warnings

from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def evaluate_clustering():
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

    X = []
    Y = []
    nb_cluster_found = []

    max_skills_sets_sizes = 30
    for i in range(3, max_skills_sets_sizes):
        print(i)
        X.append(i)
        skills_sets = generate_skills_sets(i, 5, 7)
        users_skills, clusters_ground_truth = generate_user_skills(
            skills_sets, 500, 1, 2)

        clustering_range = (3, max_skills_sets_sizes)
        clustering_model = clustering(
            users_skills, range(*clustering_range), False)

        nb_cluster_found.append(len(clustering_model.cluster_centers_))

        info_score = normalized_mutual_info_score(
            clusters_ground_truth, clustering_model.labels_)
        Y.append(info_score)

    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    plt.title("Normalized mutual info score over number of skills sets/jobs")
    plt.xlabel("Number of skill sets/jobs")
    plt.ylabel("Normalized mutual info score")
    X = np.array(X)
    Y = np.array(Y)
    plt.plot(X, Y)

    is_correct = np.array(X) == np.array(nb_cluster_found)
    correct_indices = np.where(is_correct)[0]
    incorrect_indices = np.where(np.logical_not(is_correct))[0]
    plt.scatter(X[correct_indices], Y[correct_indices],
                color="blue", label="Correct number of cluster found")
    plt.scatter(X[incorrect_indices], Y[incorrect_indices],
                color="red", label="Incorrect number of cluster found")
    plt.ylim(0.95, 1.05)
    plt.legend()
    plt.savefig("evaluation_clustering.png")
    plt.show()


def evaluate_link_prediciton():
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    clustering_range = (2, 10)
    distance_function = "euclidean"

    X = []
    ms_train = []
    ms_test = []

    skills_sets = generate_skills_sets(10, 3, 6)

    for i in [250, 500, 750, 1000, 1250, 1500, 1750]:
        print(i)
        X.append(i)
        users_skills, clusters_ground_truth = generate_user_skills(skills_sets, i, 1, 2)
        G = generate_graph(clusters_ground_truth)

        # always find the right number of cluster
        clustering_model = clustering(users_skills, [10], True)
        users_distances_to_centers = cdist(users_skills, clustering_model.cluster_centers_, metric=distance_function)

        model, y_train, predicted_train, y_test, predicted_test = link_prediction(G, users_distances_to_centers)

        m_train = evaluate_metrics(y_train, predicted_train)
        ms_train.append(m_train)
        m_test = evaluate_metrics(y_test, predicted_test)
        ms_test.append(m_test)

    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    plt.title("Metrics on link prediction over graph size")
    plt.xlabel("Graph size")
    plt.ylabel("Metrics")

    metrics = ["Precision", "Recall", "F1-Score", "Confusion matrix"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(0, 3):
        l = metrics[i]

        Y = [m[i] for m in ms_train]
        plt.plot(X, Y, linestyle="--", color=colors[i])

        Y = [m[i] for m in ms_test]
        plt.plot(X, Y, color=colors[i])

    lines = [Line2D([0], [0], linestyle="--", color="k"), Line2D([0], [0], color="k"), Patch(color=colors[0]), Patch(color=colors[1]), Patch(color=colors[2])]
    label = ["Train", "Test", "Precision", "Recall", "F1-Score"]
    plt.legend(lines, label)
    plt.savefig("evaluation_link_prediction.png")
    plt.tight_layout()
    plt.show()


def evaluate_metrics(y, predicted):
    p = precision_score(y, predicted)
    r = recall_score(y, predicted)
    f1 = f1_score(y, predicted)
    cm = confusion_matrix(y, predicted)
    return p, r, f1, cm


def print_evaluate(y, predicted):
    p, r, f1, cm = evaluate_metrics(y, predicted)
    print(f"Precision : {p:.2f}")
    print(f"Recall : {r:.2f}")
    print(f"F1-Score : {f1:.2f}")
    print("Confusion matrix: ")
    print(cm)
    return p, r, f1, cm


if __name__ == "__main__":
    np.random.seed(int(np.pi * 42))
    np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

    print("Clustering evaluation")
    evaluate_clustering()
    print("Link prediction evaluation")
    evaluate_link_prediciton()
