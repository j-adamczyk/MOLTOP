from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
import torch
import torch_geometric
from ogb.utils import smiles2graph
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
from torch_geometric.nn.aggr.fused import FusedAggregation

from feature_extraction import (
    calculate_edge_betweenness,
    calculate_adjusted_rand_index,
    calculate_scan_structural_similarity_score,
)


def graph_from_smiles(smiles: str) -> Data:
    graph_dict = smiles2graph(smiles)
    return Data(
        x=torch.from_numpy(graph_dict["node_feat"]),
        edge_index=torch.from_numpy(graph_dict["edge_index"]),
        edge_attr=torch.from_numpy(graph_dict["edge_feat"]),
    )


def create_mol_image(smiles: str, filepath: str) -> None:
    mol = MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
    d.drawOptions().padding = 0.02
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    with open(filepath, "w") as file:
        file.write(d.GetDrawingText())


def extract_features(data: Data) -> dict:
    # adapted from PyTorch Geometric
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

    features = {
        "deg": ldp_features[0],
        "deg_min": ldp_features[1],
        "deg_max": ldp_features[2],
        "deg_mean": ldp_features[3],
        "deg_std": ldp_features[4],
        "EBC": calculate_edge_betweenness(graph),
        "ARI": calculate_adjusted_rand_index(graph),
        "SCAN": calculate_scan_structural_similarity_score(graph),
    }

    return features


def create_histogram_plot(
    values_dpcc: np.ndarray, values_paclitaxel: np.ndarray, filepath: str
) -> None:
    # workaround for Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
    plotly.io.kaleido.scope.mathjax = None

    fig = go.Figure()
    for values, name in [(values_dpcc, "DPCC"), (values_paclitaxel, "Paclitaxel")]:
        fig.add_trace(
            go.Histogram(
                x=values,
                histnorm="percent",
                nbinsx=5,
                name=name,
                opacity=0.75,
            )
        )
    fig.update_layout(
        template="none",
        xaxis_title_text="Value",
        yaxis_title_text="Percentage",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=60, r=20, t=0, b=50),
        font=dict(size=18),
    )
    fig.write_image(filepath)


if __name__ == "__main__":
    plots_dir = "../plots/molecules"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Dipalmitoylphosphatidylcholine
    dpcc_smiles = (
        "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"
    )
    dppc = graph_from_smiles(dpcc_smiles)
    create_mol_image(dpcc_smiles, f"{plots_dir}/mol_DPCC.svg")

    # Paclitaxel
    paclitaxel_smiles = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    paclitaxel = graph_from_smiles(paclitaxel_smiles)
    create_mol_image(paclitaxel_smiles, f"{plots_dir}/mol_paclitaxel.svg")

    dppc_features = extract_features(dppc)
    paclitaxel_features = extract_features(paclitaxel)

    for feature_name in ["EBC", "ARI", "SCAN"]:
        create_histogram_plot(
            values_dpcc=dppc_features[feature_name],
            values_paclitaxel=paclitaxel_features[feature_name],
            filepath=f"{plots_dir}/hist_{feature_name}.pdf",
        )
