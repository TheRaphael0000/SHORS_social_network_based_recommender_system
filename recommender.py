from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from operator import mul
from functools import reduce


def link_prediction(G, users_distances_to_centers):
    X, y = get_features(G, users_distances_to_centers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # best params found using GridSearch
    best_params = {'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'n_estimators': 200}

    model = RandomForestClassifier(**best_params)

    # parameters = {
    #     "n_estimators": np.arange(50, 400, 50),
    #     "criterion": ["gini", "entropy"],
    #     "max_features": ["sqrt", "log2", None],
    #     "max_depth": np.arange(2, 10, 1),
    # }
    # print("Number of runs :", reduce(
    #     mul, [len(pl) for pl in parameters.values()], 1))
    # model = GridSearchCV(model, parameters, n_jobs=-1, scoring="f1")

    model.fit(X_train, y_train)

    predicted_train = model.predict(X_train)

    if hasattr(model, "best_params_"):
        print(model.best_params_)

    predicted_test = model.predict(X_test)
    return model, y_train, predicted_train, y_test, predicted_test


def get_features(G, users_distances_to_centers):
    pos_edges = list(G.edges)

    neg_edges = []
    for i, neg_edge in enumerate(generate_neg_edges(G)):
        neg_edges.append(neg_edge)
        if len(neg_edges) >= len(pos_edges):
            break

    edges = neg_edges + pos_edges
    y = [False] * len(neg_edges) + [True] * len(pos_edges)
    X, cluster_names = generate_links_features(
        edges, G, users_distances_to_centers)
    X = pd.DataFrame(X)
    return X, y


def predict_links(model, G, node, users_distances_to_centers):
    # every nodes - himself - it's neighbors
    possible_new_nodes = set(G.nodes) - {node} - set(G.neighbors(node))
    # create the links
    edges = [(node, possible_new_node)
             for possible_new_node in possible_new_nodes]
    # generate the according features
    X, cluster_names = generate_links_features(
        edges, G, users_distances_to_centers)
    X = pd.DataFrame(X)

    # transform the result
    predicted = np.array(model.predict(X))
    predictions = np.array(edges)[predicted][:, 1]

    # ranking the results based on the similarity

    # filter only the predictions
    p = X[predicted]
    # keep only the clusters distances columns
    p = p[cluster_names]
    # sum the columns
    p = p.sum(axis=1)

    # convert to lists
    scores = list(p)
    predictions = list(predictions)
    predictions_scores = dict(zip(predictions, scores))

    # rank the results
    predictions_sorted = sorted(
        predictions_scores, key=lambda k: predictions_scores[k])
    scores = [predictions_scores[k] for k in predictions_sorted]

    return np.array(predictions_sorted), np.array(scores)


def generate_neg_edges(G):
    pos_edges = set(G.edges)
    nodes = list(G.nodes)
    previously_found = set()
    # it can never stop if the graph has more than
    # half of edge of the complete graph
    # (really unlikely in a social graph, since it's really sparse)
    while True:
        a = np.random.choice(nodes)
        b = np.random.choice(nodes)
        # avoid yielding loop edges
        if a == b:
            continue
        # avoid yielding edge already exisiting
        if (a, b) in pos_edges or (b, a) in pos_edges:
            continue
        # avoid yielding edge already found
        if (a, b) in previously_found or (b, a) in previously_found:
            continue
        edge = (a, b)
        previously_found.add(edge)
        yield edge


def generate_link_features(edge, G, users_distances_to_centers):
    link_features = {}

    a, b = edge
    a_neighbor = set(G.neighbors(a))
    b_neighbor = set(G.neighbors(b))
    link_features["jaccard"] = jaccard(a_neighbor, b_neighbor)
    link_features["cosine"] = cosine(a_neighbor, b_neighbor)
    link_features["adar_index"] = adar_index(G, a_neighbor, b_neighbor)
    link_features["shortest_path"] = shortest_path(G, a, b)

    dist_to_centers_a = users_distances_to_centers[a]
    dist_to_centers_b = users_distances_to_centers[b]

    cluster_names = []
    for i, (dist_a, dist_b) in enumerate(zip(dist_to_centers_a, dist_to_centers_b)):
        d = (dist_b - dist_a)**2
        cluster_name = "cluster_" + str(i)
        link_features[cluster_name] = d
        cluster_names.append(cluster_name)

    return link_features, cluster_names


def generate_links_features(edges, G, users_distances_to_centers):
    features = defaultdict(lambda: [])

    for edge in edges:
        link_features, cluster_names = generate_link_features(
            edge, G, users_distances_to_centers)
        for feature_name, feature_value in link_features.items():
            features[feature_name].append(feature_value)

    return features, cluster_names


def jaccard(a, b):
    num = len(a.intersection(b))
    den = len(a.union(b))
    if den <= 0:
        return 0
    return num / den


def cosine(a, b):
    num = len(a.union(b))
    den = (len(a) * len(b))**(0.5)
    if den <= 0:
        return 0
    return num / den


def adar_index(G, a, b):
    return np.sum([1 / np.log(len(set(G.neighbors(i)))) for i in a.intersection(b)])


def shortest_path(G, a, b):
    G_copy = G.copy()
    if G_copy.has_edge(a, b):
        G_copy.remove_edge(a, b)
    try:
        d = nx.shortest_path_length(G_copy, a, b)
    except nx.NetworkXNoPath:
        d = -1
    return d
