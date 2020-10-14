import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import collections

from generate_data import generate_skills, generate_graph

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

# Data generation parameters
skills_sets = [
    {"Assembly", "C", "C++", "Rust"},  # System
    {"Java", "C#", "Go"},  # OOP
    {"Python", "R"},  # Statistics
    {"bash", "zsh", "sh", "batch"},  # Scripting / Shells
    {"JavaScript", "HTML", "CSS", "PHP"},  # Web
]
seed = 42  # Seed for random number generation
np.random.seed(seed)

N = 1000  # The number of nodes
K = 4  # Each node is connected to k nearest neighbors in ring topology
P = 0.2  # The probability of rewiring each edge

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 3  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 5  # Maximal of random edition of the user skill sets

# Generate skills
users_S = generate_skills(skills_sets, N, min_skill_sets,
                          max_skill_sets, min_edits, max_edits)


# Generate graph
G = generate_graph(N, K, P, seed)

print(users_S)

# print(G.nodes)
nx.draw(G)
plt.savefig("test.png")
