import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from generate_data import get_named_synthetic_dataset
from clustering import clustering, fzclustering
from recommender import link_prediction
from visualization import visualization
from evaluation import print_evaluate


def use_case_fuzzy_cmean(G, users_skills, clusters_ground_truth):
    clustering_range = (2, 10)
    distance_function = "euclidean"

    print("Clustering")
    print("Using Fuzzy C-Means")
    fuzzyclustering_model = fzclustering(users_skills, range(*clustering_range), True)
    # returned values with order
    # Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).
    print("- Number of clusters found", len(fuzzyclustering_model[0]))
    print("- Real number of clusters", len(np.unique(clusters_ground_truth)))

    users_distances_to_centers = cdist(
        users_skills, fuzzyclustering_model[0], metric=distance_function)

    # pca = PCA(n_components=2)
    #
    # pca.fit(users_skills)
    # new_data = pca.transform(users_skills)
    #
    # pca.fit(fuzzyclustering_model[0])
    # new_data2 = pca.transform(fuzzyclustering_model[0])
    # c = np.concatenate((fuzzyclustering_model[1], np.array([6] * len(fuzzyclustering_model[0]))))
    # new_data = np.concatenate((new_data, new_data2), axis=0)
    #
    # axs[1, 1].scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)

    # print("Plotting graph")
    #plot_graph(G, "Fuzzy_graph.png", colors=fuzzyclustering_model[1])

    print("Link prediction")
    model, y_train, predicted_train, y_test, predicted_test = link_prediction(
        G, users_distances_to_centers)

    print("Evaluation")
    print("- Train")
    print_evaluate(y_train, predicted_train)
    print("- Test")
    print_evaluate(y_test, predicted_test)

    print("Visualization")
    visualization(model, G, users_distances_to_centers, fuzzyclustering_model[1])


if __name__ == '__main__':
    np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
    np.random.seed(int(np.pi * 42))  # Seed for random number generation

    G, users_skills, clusters_ground_truth = get_named_synthetic_dataset()

    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle('Principal component analysis', fontsize=16)
    # axs[0, 0].set_ylabel('Ground Truth')
    # axs[1, 0].set_ylabel('KMeans')
    # axs[1, 1].set_ylabel('Fuzzy CMeans')
    #
    # axs[1].plot(t3, np.cos(2*np.pi*t3), '--')
    # axs[1].set_xlabel('time (s)')
    # axs[1].set_title('subplot 2')
    # axs[1].set_ylabel('Undamped')

    #Principal component analysis for ground Truth
    # pca = PCA(n_components=2)
    # pca.fit(users_skills)
    # new_data = pca.transform(users_skills)
    # axs[0, 0].scatter(new_data.T[0], new_data.T[1].T, c=clusters_ground_truth, alpha=0.5)

    use_case_fuzzy_cmean(G, users_skills, clusters_ground_truth)
