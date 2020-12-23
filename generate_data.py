import numpy as np
import networkx as nx
import secrets


def generate_skills_sets(nb_skills_sets, skill_sets_min_size, skill_sets_max_size):
    """Generate skill sets
    nb_skills_sets: The number of skills sets to create (jobs)
    skill_sets_min_size: The minimal number of skills in each set
    skill_sets_max_size: The maximal number of skills in a set
    """
    skills_sets = []
    for i in range(nb_skills_sets):
        skills_set = []
        for j in range(np.random.randint(skill_sets_min_size, skill_sets_max_size)):
            skills_set.append(secrets.token_urlsafe(4))
        skills_sets.append(skills_set)
    return skills_sets


def generate_user_skills(skills_sets, N, min_edits, max_edits):
    """Generate N random users_skills based on skills_sets
    add a bit of noise with the min and max edits
    """
    all_skills = list()
    for ss in skills_sets:
        all_skills += ss

    users_skills = []
    clusters_ground_truth = []

    for _ in range(N):
        user_skills = np.zeros((len(all_skills)), dtype=bool)

        skills_sets_indice = np.random.randint(len(skills_sets))

        skill_set = skills_sets[skills_sets_indice]
        for skill in skill_set:
            i = all_skills.index(skill)
            user_skills[i] = True

        nb_edits = np.random.randint(min_edits, max_edits)
        for _ in range(nb_edits):
            # flip a random bit
            if np.random.rand() >= 0.5:
                # skill to add
                false_indices = np.where(np.logical_not(user_skills))[0]
                if len(false_indices) <= 0:
                    continue
                i = np.random.choice(false_indices)
                # flip the bit
                user_skills[i] ^= True
            else:
                # remove a skill
                true_indices = np.where(user_skills)[0]
                if len(true_indices) <= 0:
                    continue
                i = np.random.choice(true_indices)
                user_skills[i] ^= True


        users_skills.append(user_skills)
        clusters_ground_truth.append(skills_sets_indice)

    users_skills = np.array(users_skills)
    clusters_ground_truth = np.array(clusters_ground_truth)

    return users_skills, clusters_ground_truth


def generate_graph(clusters_ground_truth, cluster_boost=3, m=2):
    """Creating a graph according to the PREFERENTIAL ATTACHMENT MODEL
    for a social graph alike"""
    G = nx.Graph()

    # initialize the two first users
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1)

    for c_node, cluster in list(enumerate(clusters_ground_truth))[2:]:
        candidates = list(G.nodes)
        G.add_node(c_node)

        degrees = np.array([G.degree[node] for node in candidates])
        P_degrees = degrees / degrees.sum()
        # prefer to attach to people in it's own cluster
        P_cluster = np.array([cluster_boost if clusters_ground_truth[node]
                              == cluster else 1 / cluster_boost for node in candidates])
        P = P_degrees * P_cluster

        while G.degree[c_node] < m:
            potential_node = np.random.randint(0, len(candidates))
            p = P[potential_node]
            if np.random.random() <= p:
                G.add_edge(c_node, potential_node)
            candidates = np.delete(candidates, potential_node)
            P = np.delete(P, potential_node)
            if len(candidates) <= 0:
                break

    return G
