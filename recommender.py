import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve


def link_prediction(G, users_skills):
    pos_edges = list(G.edges)
    neg_edges = generate_neg_edges(G)

    edges = neg_edges + pos_edges
    y = [-1] * len(neg_edges) + [1] * len(pos_edges)
    X = generate_features(G, edges)
    X = pd.DataFrame(X)

    rfc = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    print("- Fitting")
    rfc.fit(X_train, y_train)

    print("n_features_", rfc.n_features_)

    print("- Predicting")
    predicted = rfc.predict(X_test)
    print(confusion_matrix(y_test, predicted))
    precision, recall, thresholds = precision_recall_curve(y_test, predicted)


def generate_neg_edges(G):
    pos_edges = set(G.edges)
    nodes = list(G.nodes)
    neg_edges = set()
    while len(neg_edges) < len(pos_edges):
        a = np.random.choice(nodes)
        b = np.random.choice(nodes)
        # avoid adding
        if a == b or (a, b) in pos_edges or (b, a) in pos_edges:
            continue
        neg_edges.add((a, b))
    return list(neg_edges)


def generate_features(G, edges):

    features = {}

    for f in ["jaccard", "cosine", "shortest_path", "adar_index"]:
        features[f] = []

    for a, b in edges:
        a_neighbor = set(G.neighbors(a))
        b_neighbor = set(G.neighbors(b))
        features["jaccard"].append(jaccard(a_neighbor, b_neighbor))
        features["cosine"].append(cosine(a_neighbor, b_neighbor))
        features["adar_index"].append(adar_index(G, a_neighbor, b_neighbor))
        features["shortest_path"].append(shortest_path(G, a, b))
    return features


def jaccard(a, b):
    return len(a.intersection(b)) / len(a.union(b))


def cosine(a, b):
    return len(a.union(b)) / (len(a) * len(b))**(0.5)


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
