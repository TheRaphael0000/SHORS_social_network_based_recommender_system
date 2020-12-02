import numpy as np
from scipy.spatial.distance import dice, cdist

from generate_data import generate_skills, generate_graph
from clustering import clustering, evaluate_clustering
from misc import plot_graph
from recommender import predict_links, link_prediction


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

seed = int(np.pi * 42)  # Seed for random number generation
np.random.seed(seed)

N = 400  # The number of nodes

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
clustering_model = clustering(users_skills, range(2, 10), True)

nb_clusters_found = len(clustering_model.cluster_centers_)
print("Number of clusters found", nb_clusters_found)
print("Real number of clusters", len(skills_sets))

# Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
users_distances_to_centers = cdist(users_skills, clustering_model.cluster_centers_, metric="euclidean")

evaluate_clustering(clusters_ground_truth, clustering_model.labels_)

print("Plotting graph")
print(len(G.edges))
plot_graph(G, "graph.png", colors=clustering_model.labels_)

print("Link prediction")
link_prediction_model = link_prediction(G, users_distances_to_centers)

predictions = predict_links(link_prediction_model,
                            G, 0, users_distances_to_centers)
print(predictions)
