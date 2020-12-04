import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_node(0, label="0")
G.add_node(1, label="1")
G.add_node(2, label="2")

G.add_edge(1, 2)

G2 = nx.Graph()
for e in nx.non_edges(G):
    G2.add_edge(*e)

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos=pos)
nx.draw_networkx_labels(G, pos, {0: "1", 1: "2", 2: "3"})
nx.draw_networkx_edges(G, pos=pos, edge_color="blue")
nx.draw_networkx_edges(G2, pos=pos, edge_color="red")
plt.tight_layout()
plt.axis("off")
plt.savefig("schema.png")
