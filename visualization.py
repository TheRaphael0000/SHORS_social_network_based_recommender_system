import matplotlib.pyplot as plt
import networkx as nx
from recommender import predict_links


def visualization(link_prediction_model, G, users_distances_to_centers, colors):
    fig, ax = plt.subplots(figsize=(9, 9))
    pos = nx.spring_layout(G)
    plt.tight_layout()
    plt.axis("off")

    def draw():
        plt.clf()
        #nx.draw_networkx_nodes(G, pos=pos)
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors,
                               cmap=plt.get_cmap("Paired"))
        nx.draw_networkx_edges(G, pos=pos)

    def onclick(event):
        if not event.dblclick:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        dists = {k: (v[0] - x)**2 + (v[1] - y)**2 for k, v in pos.items()}
        closest_node = min(dists, key=dists.get)

        predictions, scores = predict_links(
            link_prediction_model, G, closest_node, users_distances_to_centers)
        # edgelist = [(closest_node, p) for p in predictions]

        top_no = 10
        top = predictions[0:top_no]

        print(f"Top {top_no} recommendations for user {closest_node} : {top}")

        xlim = plt.xlim()
        ylim = plt.ylim()
        draw()
        # draw the predicted links with they rank as alpha
        for i, p in enumerate(top):
            Xs = [pos[closest_node][0], pos[p][0]]
            Ys = [pos[closest_node][1], pos[p][1]]
            alpha = (len(top) - i) / len(top)
            plt.plot(Xs, Ys, color="red", alpha=alpha)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    draw()
    plt.show()
