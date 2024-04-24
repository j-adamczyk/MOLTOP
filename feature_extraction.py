import networkit
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils
from networkit.centrality import Betweenness
from networkit.linkprediction import AdjustedRandIndex
from networkit.sparsification import TriangleEdgeScore, SCANStructuralSimilarityScore
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.aggr.fused import FusedAggregation
from tqdm import tqdm


def extract_features(
    dataset: Dataset,
    degree_features: bool,
    edge_betweenness: bool,
    rand_index: bool,
    scan_structural_score: bool,
    atom_types: bool,
    bond_types: bool,
    n_bins: int,
    verbose: bool,
) -> np.ndarray:
    if verbose:
        print("Extracting features")

    iterable = tqdm(dataset) if verbose else dataset
    rows = [
        process_graph(
            data,
            degree_features=degree_features,
            edge_betweenness=edge_betweenness,
            rand_index=rand_index,
            scan_structural_score=scan_structural_score,
            atom_types=atom_types,
            bond_types=bond_types,
            n_bins=n_bins,
        )
        for data in iterable
    ]

    X = np.stack(rows)
    return X


def process_graph(
    data: Data,
    degree_features: bool,
    edge_betweenness: bool,
    rand_index: bool,
    scan_structural_score: bool,
    atom_types: bool,
    bond_types: bool,
    n_bins: int,
) -> np.array:
    row, col = data.edge_index
    num_nodes = data.num_nodes

    deg = torch_geometric.utils.degree(row, num_nodes, dtype=torch.float)
    deg = deg.view(-1, 1)
    deg_col = deg[col]

    aggr = FusedAggregation(["min", "max", "mean", "std"])
    ldp_features = [deg] + aggr(deg_col, row, dim_size=num_nodes)
    ldp_features = [feature.numpy().ravel() for feature in ldp_features]

    graph = torch_geometric.utils.to_networkit(
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        num_nodes=data.num_nodes,
        directed=False,
    )
    graph.indexEdges()

    topological_descriptors = []
    if edge_betweenness:
        value = calculate_edge_betweenness(graph)
        topological_descriptors.append(value)
    if rand_index:
        value = calculate_adjusted_rand_index(graph)
        topological_descriptors.append(value)
    if scan_structural_score:
        value = calculate_scan_structural_similarity_score(graph)
        topological_descriptors.append(value)

    molecular_features = []

    if atom_types:
        # MoleculeNet OGB featurization - atomic number is the first feature
        atom_features = data.x[:, 0]
        atom_features = F.one_hot(atom_features, 120).float()
        atom_features = atom_features[:, :89]

        atom_types_mean = torch.mean(atom_features, dim=0).numpy()
        atom_types_std = torch.std(atom_features, dim=0).numpy()
        atom_types_sum = torch.sum(atom_features, dim=0).numpy()

        # in case of all-zero features standard deviation is NaN, we fill it with zeros
        atom_types_std[np.isnan(atom_types_std)] = 0

        atom_type_features = np.concatenate(
            (
                atom_types_mean,
                atom_types_std,
                atom_types_sum,
            )
        )
        molecular_features.append(atom_type_features)

    if bond_types:
        # MoleculeNet OGB featurization - bond type is the first feature
        bond_features = data.edge_attr[:, 0]
        bond_features = F.one_hot(bond_features, 5).float()

        # there are a few graphs without edge features, we use all zeros
        if bond_features.shape[0] == 0:
            bond_features = np.zeros(15, dtype=float)
        else:
            bond_features_mean = torch.mean(bond_features, dim=0).numpy()
            bond_features_std = torch.std(bond_features, dim=0).numpy()
            bond_features_sum = torch.sum(bond_features, dim=0).numpy()

            # in case of all-zero features standard deviation is NaN, we fill it with zeros
            bond_features_std[np.isnan(bond_features_std)] = 0

            bond_features = np.concatenate(
                (
                    bond_features_mean,
                    bond_features_std,
                    bond_features_sum,
                )
            )

        molecular_features.append(bond_features)

    # aggregate all features with histograms
    topological_features = []

    if degree_features:
        for feature in ldp_features[:3]:
            values = np.bincount(feature.astype(int), minlength=11)[:11]
            values = values.astype(float)
            topological_features.append(values)

        for feature in ldp_features[3:]:
            values, _ = np.histogram(feature, bins=n_bins)
            topological_features.append(values)

    if edge_betweenness or rand_index or scan_structural_score:
        for feature in topological_descriptors:
            values, _ = np.histogram(feature, bins=n_bins)
            topological_features.append(values)

    features = np.concatenate(topological_features + molecular_features)
    return features


def calculate_edge_betweenness(graph: networkit.Graph) -> np.ndarray:
    betweeness = Betweenness(graph, normalized=True, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    scores = np.array(scores, dtype=float)
    return scores


def calculate_adjusted_rand_index(graph: networkit.Graph) -> np.ndarray:
    index = AdjustedRandIndex(graph)
    scores = [index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=float)
    return scores


def calculate_scan_structural_similarity_score(graph: networkit.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = SCANStructuralSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    scores = np.array(scores, dtype=float)
    return scores
