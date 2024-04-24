import itertools
import math

import networkx as nx
import numpy as np

from expressivity.utils import extract_features, get_feature_from_vector


if __name__ == "__main__":
    graphs: list[nx.Graph] = nx.read_graph6("data/sr251256.g6")
    n_bins = 25  # median - in SR25, every graph has 25 nodes
    features = [extract_features(graph, n_bins) for graph in graphs]

    correct = 0
    total = math.comb(len(graphs), 2)

    for name in ["LDP_degrees", "EBC", "ARI", "SCAN", "all features"]:
        correct = 0
        for i, j in itertools.combinations(list(range(len(graphs))), 2):
            if name != "all features":
                G1_values = get_feature_from_vector(features[i], name, n_bins)
                G2_values = get_feature_from_vector(features[j], name, n_bins)
            else:
                G1_values = features[i]
                G2_values = features[j]
            if not np.all(np.equal(G1_values, G2_values)):
                correct += 1

        print(f"Feature {name}")
        print(f"Correct count: {correct}")
        print(f"Accuracy: {100 * correct / total:.2f}%")
        print()
