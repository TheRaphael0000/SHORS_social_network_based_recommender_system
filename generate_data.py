import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import collections
import random


def generate_skills(skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits):
    users_skills = []

    for i in range(N):

        nb_skill_sets = np.random.randint(min_skill_sets, max_skill_sets)
        skills_sets_indices = np.random.choice(
            range(len(skills_sets)), nb_skill_sets)

        user_skills = []
        other_skills = []
        for i, s in enumerate(skills_sets):
            if i in skills_sets_indices:
                user_skills.extend(list(s))
            else:
                other_skills.extend(list(s))

        nb_edits = np.random.randint(min_edits, max_edits)
        for e in range(nb_edits):

            def add():
                new_skill = np.random.choice(other_skills)
                other_skills.remove(new_skill)
                user_skills.append(new_skill)

            def remove():
                skill_to_remove = np.random.choice(user_skills)
                user_skills.remove(skill_to_remove)
                other_skills.append(skill_to_remove)

            if len(user_skills) <= 0:
                add()
            elif len(other_skills) <= 0:
                remove()
            else:
                r = np.random.random()
                if r > 0.5:
                    add()
                else:
                    remove()

        users_skills.append(user_skills)

    return users_skills


def generate_graph(N, K, P, seed):
    G = nx.watts_strogatz_graph(N, K, P, seed)
    return G
