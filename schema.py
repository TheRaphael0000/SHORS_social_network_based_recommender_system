# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt

G = nx.frucht_graph()

G2 = nx.Graph()
for e in nx.non_edges(G):
    G2.add_edge(*e)

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos=pos)
# nx.draw_networkx_labels(G, pos, {0: "1", 1: "2", 2: "3"})
nx.draw_networkx_edges(G2, pos=pos, edge_color="red")
nx.draw_networkx_edges(G, pos=pos, edge_color="blue")
plt.tight_layout()
plt.axis("off")
plt.savefig("schema.png")
