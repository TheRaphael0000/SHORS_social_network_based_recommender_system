import numpy as np
from scipy.spatial.distance import dice, cdist

from generate_data import generate_skills, generate_graph
from clustering import clustering, evaluate_clustering, fzclustering
from recommender import predict_links, link_prediction
from visualization import visualization
from misc import plot_graph

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

# Data generation parameters
skills_sets = [
    ["Assembly", "C", "C++", "Rust"],  # System
    ["Java", "C#", "Go"],  # OOP
    ["Python", "R"],  # Statistics
    ["bash", "zsh", "sh", "batch"],  # Scripting / Shells
    ["JavaScript", "HTML", "CSS", "PHP"],  # Web
    ["SAP", "Microsoft Dynamics", "Odoo", "Spreadsheet"],  # Management
]

seed = int(np.pi * 37)  # Seed for random number generation
np.random.seed(seed)

N = 100  # The number of nodes

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

set_distance_function = dice

print("Generating skills")
users_skills, clusters_ground_truth = generate_skills(
    skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

print("Generating graph")
G = generate_graph(clusters_ground_truth)

print("Clustering")
clustering_model = clustering(users_skills, range(2, 7), True)
# returned values with order
# Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).
fuzzyclustering_model = fzclustering(users_skills, range(2, 7), True)

nb_clusters_found = len(clustering_model.cluster_centers_)
fuzzy_nb_clusters_found = len(fuzzyclustering_model[1])

print("- Number of clusters found", nb_clusters_found)
print("- Number of fuzzy clusters found", fuzzy_nb_clusters_found)
print("- Real number of clusters", len(skills_sets))

# Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
users_distances_to_centers = cdist(users_skills, clustering_model.cluster_centers_, metric="euclidean")
users_distances_to_centers_fuzzy = cdist(users_skills, fuzzyclustering_model[1], metric="euclidean")

evaluate_clustering(clusters_ground_truth, clustering_model.labels_)


# print("Plotting graph")
plot_graph(G, "graph.png", colors=clustering_model.labels_)
#plot_graph(G, "graph_fuzzy.png", colors=fuzzyclustering_model[7])


print("Link prediction")
link_prediction_model = link_prediction(G, users_distances_to_centers)
link_prediction_model_fuzzy = link_prediction(G, users_distances_to_centers_fuzzy)

# predictions, scores = predict_links(link_prediction_model,
#                             G, 0, users_distances_to_centers)
# print(predictions)
# print(scores)

visualization(link_prediction_model, G, users_distances_to_centers, clustering_model.labels_)
visualization(link_prediction_model_fuzzy, G, users_distances_to_centers_fuzzy, None)

