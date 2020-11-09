import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import dice

from generate_data import generate_skills, generate_graph
from similarity import skills_similarity, user_similarity
from clustering import clustering, get_distance_to_center


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

N = 1000  # The number of nodes
K = 4  # Each node is connected to k nearest neighbors in ring topology
P = 0.2  # The probability of rewiring each edge

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

set_distance_function = dice

# Generate skills
users_skills = generate_skills(
    all_skills, skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)


model = clustering(users_skills, range(2, 15), True)
users_distances_to_centers = get_distance_to_center(
    users_skills, model.cluster_centers_)

# Generate graph
G = generate_graph(N, K, P, seed)

skills_similarity_matrix = skills_similarity(all_skills, users_skills)

user_similarity_matrix = user_similarity(
    all_skills, users_skills, set_distance_function)

print(skills_similarity_matrix)
print(user_similarity_matrix)

# print(G.nodes)
nx.draw(G)
plt.savefig("graph.png")
