import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.utils import from_networkx

from feature_extraction import (
    calculate_edge_betweenness,
    calculate_adjusted_rand_index,
    calculate_scan_structural_similarity_score,
)


def extract_features(nx_graph: nx.Graph, n_bins: int) -> np.array:
    pyg_graph = from_networkx(nx_graph)
    nk_graph = torch_geometric.utils.to_networkit(
        edge_index=pyg_graph.edge_index,
        edge_weight=pyg_graph.edge_weight,
        num_nodes=pyg_graph.num_nodes,
        directed=False,
    )
    nk_graph.indexEdges()

    row, col = pyg_graph.edge_index
    num_nodes = pyg_graph.num_nodes

    deg = torch_geometric.utils.degree(row, num_nodes, dtype=torch.float)
    deg = deg.view(-1, 1)
    deg_col = deg[col]

    aggr = FusedAggregation(["min", "max", "mean", "std"])
    ldp_features = [deg] + aggr(deg_col, row, dim_size=num_nodes)
    ldp_features = [feature.numpy().ravel() for feature in ldp_features]

    ebc = calculate_edge_betweenness(nk_graph)
    ari = calculate_adjusted_rand_index(nk_graph)
    scan = calculate_scan_structural_similarity_score(nk_graph)

    features = []
    for feature in ldp_features + [ebc, ari, scan]:
        values, _ = np.histogram(feature.astype(float), bins=n_bins)
        features.append(values)

    return np.concatenate(features)


def get_feature_from_vector(
    embedding: np.ndarray, feature: str, n_bins: int
) -> np.ndarray:
    if feature == "LDP_degrees":
        return embedding[: n_bins * 5]
    elif feature == "EBC":
        return embedding[n_bins * 5 : n_bins * 6]
    elif feature == "ARI":
        return embedding[n_bins * 6 : n_bins * 7]
    elif feature == "SCAN":
        return embedding[n_bins * 7 :]
    else:
        raise ValueError(f"Value '{feature}' not recognized")
