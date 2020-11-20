import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import dice
from scipy.spatial.distance import cdist

from generate_data import generate_skills, generate_graph
from similarity import skills_similarity, user_similarity
from clustering import clustering

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
all_skills = list()
for ss in skills_sets:
    all_skills += ss

seed = 42  # Seed for random number generation
np.random.seed(seed)

N = 200  # The number of nodes
K = 4  # Each node is connected to k nearest neighbors in ring topology
P = 0.2  # The probability of rewiring each edge

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

set_distance_function = dice

print("Generating skills")
users_skills = generate_skills(
    all_skills, skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

print("Generating graph")
G = generate_graph(N, K, P, seed)

print("Clustering")
model = clustering(users_skills, range(2, 10), True)
# Possible distances metrics : "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule".
# users_distances_to_centers = cdist(
#     users_skills, model.cluster_centers_, metric="cosine")
nb_clusters_found = len(model.cluster_centers_)
print("Number of clusters found", nb_clusters_found)

zero_nodes = [n for n in G.nodes if model.labels_[n] == 0]
G_Zero = G.subgraph(zero_nodes)

from networkx.algorithms.community.centrality import girvan_newman
g = girvan_newman(G)
comp = None
i = 0
for comp in g:
    i += 1
    if i > 1:
        break
colors = []
for i in range(N):
    if i in comp[0]:
        colors.append(0)
    if i in comp[1]:
        colors.append(1)
    if i in comp[2]:
        colors.append(2)
print(comp)
print(colors)

print("Plotting graph")

# print(G.nodes)

# plot_graph(G_Zero, colors=[0] * len(G_Zero.nodes))
plot_graph(G, colors=colors)
