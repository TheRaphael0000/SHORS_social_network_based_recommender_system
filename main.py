import numpy as np
from scipy.spatial.distance import cdist

from generate_data import generate_skills, generate_graph
from clustering import clustering, evaluate_clustering, fzclustering
from recommender import link_prediction
from visualization import visualization
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

# Data generation parameters
skills_sets = [
    ["Assembly", "C", "C++", "Rust"],  # System
    ["JavaScript", "HTML", "CSS", "PHP"],  # Web
    ["Java", "C#", "Go"],  # OOP
    ["bash", "zsh", "sh", "batch"],  # Scripting / Shells
    ["Python", "R"],  # Statistics
    ["SAP", "Microsoft Dynamics", "Odoo", "Spreadsheet"],  # Management
]

seed = int(np.pi * 42)  # Seed for random number generation
np.random.seed(seed)

use_fuzzy_clustering = True

N = 500  # The number of nodes

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

# Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
clustering_range = (2, 10)
distance_function = "euclidean"

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Principal component analysis', fontsize=16)
axs[0, 0].set_ylabel('Ground Truth')
axs[1, 0].set_ylabel('KMeans')
axs[1, 1].set_ylabel('Fuzzy CMeans')


#axs[1].plot(t3, np.cos(2*np.pi*t3), '--')
#axs[1].set_xlabel('time (s)')
#axs[1].set_title('subplot 2')
#axs[1].set_ylabel('Undamped')


def use_case_fuzzy_cmean(users_skills, clusters_ground_truth):
    print("Clustering")
    print("Using Fuzzy C-Means")
    fuzzyclustering_model = fzclustering(users_skills, range(*clustering_range), True)
    # returned values with order
    # Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).
    print("- Number of clusters found", len(fuzzyclustering_model[0]))
    print("- Real number of clusters", len(skills_sets))

    users_distances_to_centers = cdist(
        users_skills, fuzzyclustering_model[0], metric=distance_function)
    print(evaluate_clustering(clusters_ground_truth, fuzzyclustering_model[1]))

    pca = PCA(n_components=2)
    #
    pca.fit(users_skills)
    new_data = pca.transform(users_skills)
    #
    pca.fit(fuzzyclustering_model[0])
    new_data2 = pca.transform(fuzzyclustering_model[0])
    c = np.concatenate((fuzzyclustering_model[1], np.array([6] * len(fuzzyclustering_model[0]))))
    new_data = np.concatenate((new_data, new_data2), axis=0)
    #
    axs[1, 1].scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)

    # print("Plotting graph")
    #plot_graph(G, "Fuzzy_graph.png", colors=fuzzyclustering_model[1])

    print("Link prediction")
    link_prediction_model_fuzzy = link_prediction(
        G, users_distances_to_centers)

    print("Visualization")
    visualization(link_prediction_model_fuzzy, G, users_distances_to_centers, fuzzyclustering_model[1])


def use_case_kmeans(users_skills, clusters_ground_truth):
    print("Clustering")
    print("Using KMeans")
    clustering_model = clustering(users_skills, range(*clustering_range), True)
    print("- Number of clusters found", len(clustering_model.cluster_centers_))
    print("- Real number of clusters", len(skills_sets))

    users_distances_to_centers = cdist(
        users_skills, clustering_model.cluster_centers_, metric=distance_function)
    evaluate_clustering(clusters_ground_truth, clustering_model.labels_)

    pca = PCA(n_components=2)
    #
    pca.fit(users_skills)
    new_data = pca.transform(users_skills)
    #
    pca.fit(clustering_model.cluster_centers_)
    new_data2 = pca.transform(clustering_model.cluster_centers_)
    c = np.concatenate((clustering_model.labels_, np.array([6] * 6)))
    new_data = np.concatenate((new_data, new_data2), axis=0)
    #
    axs[1, 0].scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)

    # print("Plotting graph")
    #plot_graph(G, "KMeans_graph.png", colors=clustering_model.labels_)

    print("Link prediction")
    link_prediction_model = link_prediction(G, users_distances_to_centers)

    print("Visualization")
    visualization(link_prediction_model, G,
                  users_distances_to_centers, clustering_model.labels_)


if __name__ == '__main__':
    print("Generating skills")
    users_skills, clusters_ground_truth = generate_skills(
        skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)
    print("Generating graph")
    G = generate_graph(clusters_ground_truth)

    # Principal component analysis for ground Truth
    pca = PCA(n_components=2)
    pca.fit(users_skills)
    new_data = pca.transform(users_skills)
    axs[0, 0].scatter(new_data.T[0], new_data.T[1].T, c=clusters_ground_truth, alpha=0.5)

    use_case_kmeans(users_skills, clusters_ground_truth)

    use_case_fuzzy_cmean(users_skills, clusters_ground_truth)

    plt.show()
