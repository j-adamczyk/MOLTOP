import itertools
import math

import networkx as nx
import numpy as np
from tqdm import tqdm

from expressivity.utils import extract_features, get_feature_from_vector


if __name__ == "__main__":
    graphs: list[nx.Graph] = nx.read_graph6("data/graph8c.g6")
    n_bins = 8  # median - in graph8c, every graph has 8 nodes
    features = [extract_features(graph, n_bins) for graph in tqdm(graphs)]

    iterable = list(range(len(graphs)))
    total = math.comb(len(graphs), 2)

    for name in ["LDP_degrees", "EBC", "ARI", "SCAN", "all features"]:
        correct = 0
        for i, j in tqdm(itertools.combinations(iterable, 2), total=total):
            if name != "all features":
                G1_values = get_feature_from_vector(features[i], name, n_bins)
                G2_values = get_feature_from_vector(features[j], name, n_bins)
            else:
                G1_values = features[i]
                G2_values = features[j]

            if not np.all(np.equal(G1_values, G2_values)):
                correct += 1

        print(f"Feature: {name}")
        print(f"Errors count: {total - correct}")
        print()
