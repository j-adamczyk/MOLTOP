import networkx as nx
import numpy as np

from expressivity.utils import (
    extract_features,
    get_feature_from_vector,
)

"""
Small graphs from "A Survey on The Expressive Power of Graph Neural Networks" R. Sato
that cannot be distinguished by set WL-3 test, but can be distinguished by WL-3 test.

This requires e.g. detecting 4-cycles. Our method cannot distinguish such graphs.
"""


def create_first_graph() -> nx.Graph:
    """
    0 - 1   4 - 5
    |   |   |   |
    2 - 3   6 - 7
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(8)))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7)])
    return G


def create_second_graph() -> nx.Graph:
    """
    0 - 1 - 2 - 3
    |           |
    4 - 5 - 6 - 7
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(8)))
    G.add_edges_from([(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 7)])
    return G


if __name__ == "__main__":
    G1 = create_first_graph()
    G2 = create_second_graph()
    n_bins = G1.number_of_nodes()  # G1 and G2 have the same size

    G1_features = extract_features(G1, n_bins=n_bins)
    G2_features = extract_features(G2, n_bins=n_bins)

    feature_names = ["LDP_degrees", "EBC", "ARI", "SCAN"]

    for name in feature_names:
        G1_values = get_feature_from_vector(G1_features, name, n_bins)
        G2_values = get_feature_from_vector(G2_features, name, n_bins)
        distinguished = not np.all(np.equal(G1_values, G2_values))

        print(f"Feature: {name}")
        print(f"Distinguished: {distinguished}")
        print()

    distinguished = not np.all(np.equal(G1_features, G2_features))

    print("All features")
    print(f"Distinguished: {distinguished}")
