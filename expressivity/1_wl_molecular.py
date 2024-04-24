import networkx as nx
import numpy as np

from expressivity.utils import (
    extract_features,
    get_feature_from_vector,
)

"""
Example molecules, Decalin and Bicyclopentyl, from:
"A Survey on The Expressive Power of Graph Neural Networks" R. Sato
See Figure 6 in the paper. Those graphs are not isomorphic or regular.

They are also used for GSN model in:
"Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting" G. Bouritsas et a.
See Figure 2 in the paper.

Those cannot be distinguished by WL-1 test.
"""


def create_decalin_graph() -> nx.Graph:
    r"""
         0     1
       /   \ /   \
       2    3    4
       |    |    |
       5    6    7
       \   / \  /
         8     9
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(
        [
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (7, 9),
        ]
    )
    return G


def create_bicyclopentyl_graph() -> nx.Graph:
    r"""
    0 - 1       2 - 3
    |    \     /    |
    |     4 - 5     |
    |    /     \    |
    6 - 7       8 - 9
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    G.add_edges_from(
        [
            (0, 1),
            (0, 6),
            (1, 4),
            (2, 3),
            (2, 5),
            (3, 9),
            (4, 5),
            (4, 7),
            (5, 8),
            (6, 7),
            (8, 9),
        ]
    )
    return G


if __name__ == "__main__":
    G1 = create_decalin_graph()
    G2 = create_bicyclopentyl_graph()
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
