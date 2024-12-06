import pytest
import networkx as nx
import numpy as np


@pytest.fixture(scope="module")
def path_to_sol():
    def fun(path, edges):
        path_edges = list(zip(path[:-1], path[1:]))
        sol = [1 if e in path_edges else 0 for e in edges]
        return sol

    return fun


@pytest.fixture(scope="module")
def random_graph():
    G = nx.fast_gnp_random_graph(10, 0.1, directed=True, seed=123)
    node_pos = nx.spring_layout(G)
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


@pytest.fixture(scope="module")
def multigraph():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, cost=1.0, sp=True)
    G.add_edge(0, 1, cost=2.0, sp=False)
    G.add_edge(1, 2, cost=1.0, sp=True)
    G.add_edge(1, 2, cost=2.0, sp=False)
    return G
