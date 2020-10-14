import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import collections
import random


def skills_similarity(all_skills, users_skills):
    all_skills = list(all_skills)
    mat = np.zeros((len(all_skills), len(all_skills)))

    for user_skill in users_skills:
        user_skills_index = [i for i, s in enumerate(user_skill) if s]
        for a, b in itertools.combinations(user_skills_index, 2):
            mat[a, b] += 1
            mat[b, a] += 1

    return mat / len(users_skills)


def user_similarity(all_skills, users_skills, distance_function):
    mat = np.zeros((len(users_skills), len(users_skills)))

    for a, b in itertools.combinations(range(len(users_skills)), 2):
        d = distance_function(users_skills[a], users_skills[b])
        mat[a, b] = d
        mat[b, a] = d

    return mat
