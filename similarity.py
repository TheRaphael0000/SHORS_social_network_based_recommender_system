import numpy as np
import itertools


def skills_similarity(all_skills, users_skills):
    all_skills = list(all_skills)
    mat = np.zeros((len(all_skills), len(all_skills)))

    for user_skill in users_skills:
        user_skills_index = [i for i, s in enumerate(user_skill) if s]
        for a, b in itertools.combinations(user_skills_index, 2):
            mat[a, b] += 1
            mat[b, a] += 1

    return mat / len(users_skills)
