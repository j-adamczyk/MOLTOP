from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt


def create_decaprismane() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(list(range(20)))
    for i in range(20):
        if i not in [9, 19]:
            graph.add_edge(i, i + 1)
        if i + 10 < 20:
            graph.add_edge(i, i + 10)

    graph.add_edge(9, 0)
    graph.add_edge(19, 10)

    nx.set_node_attributes(graph, "C", name="element")
    return graph


def create_dodecahedrane() -> nx.Graph:
    graph: nx.Graph = nx.dodecahedral_graph()
    nx.set_node_attributes(graph, "C", name="element")
    return graph


def create_graph_plot(
    graph: nx.Graph, plots_dir: str, file_name: str, seed: int
) -> None:
    elements = nx.get_node_attributes(graph, name="element")
    nx.draw(
        graph,
        with_labels=True,
        labels=elements,
        node_color="black",
        font_color="white",
        pos=nx.spring_layout(graph, seed=seed),  # seed for nice plot result
    )
    plt.gca().set_aspect("equal")
    plt.savefig(f"{plots_dir}/{file_name}")
    plt.clf()


if __name__ == "__main__":
    plots_dir = "../plots/molecules"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Decaprismane
    graph = create_decaprismane()
    create_graph_plot(graph, plots_dir, "mol_decaprismane.pdf", seed=1)

    # Dodecahedrane
    graph = create_dodecahedrane()
    create_graph_plot(graph, plots_dir, "mol_dodecahedrane.pdf", seed=11)
