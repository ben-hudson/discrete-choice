import networkx as nx
import numpy as np
import pytest

from data_generator.shortestpath import PathChoice


@pytest.fixture
def random_graph():
    G = nx.fast_gnp_random_graph(100, 0.1, directed=True)
    node_pos = nx.spring_layout(G)
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


@pytest.fixture
def multigraph():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, cost=1.0, sp=True)
    G.add_edge(0, 1, cost=2.0, sp=False)
    G.add_edge(1, 2, cost=1.0, sp=True)
    G.add_edge(1, 2, cost=2.0, sp=False)
    return G


def path_to_sol(path, edges):
    path_edges = list(zip(path[:-1], path[1:]))
    sol = [1 if e in path_edges else 0 for e in edges]
    return np.array(sol)


def test_ilp_sp(random_graph: nx.DiGraph):
    def util(costs, n_samples, seed):
        return np.broadcast_to(-costs, (n_samples, *costs.shape))

    choice_model = PathChoice(random_graph, util_fn=util)

    cost_dict = nx.get_edge_attributes(random_graph, "cost")
    cost = np.array(list(cost_dict.values()))

    n_samples = 10
    util, sols, objs, other = choice_model.sample(cost, n_samples)
    ods = other["ods"]

    for sol, obj, (o, d) in zip(sols, objs, ods):
        true_obj, true_path = nx.single_source_dijkstra(random_graph, o, d, weight="cost")
        true_sol = path_to_sol(true_path, random_graph.edges)
        assert np.isclose(-obj, true_obj), "model and true objective values differ"
        assert (sol == true_sol).all(), "model and true paths differ"


def test_multigraph_ilp_sp(multigraph: nx.MultiDiGraph):
    def util(costs, n_samples, seed):
        return np.broadcast_to(-costs, (n_samples, *costs.shape))

    choice_model = PathChoice(multigraph, util_fn=util)

    source = 0
    sink = 2

    cost_dict = nx.get_edge_attributes(multigraph, "cost")
    cost = np.array(list(cost_dict.values()))

    util, sols, objs, other = choice_model.sample(cost, 1, source, sink)
    sol = sols.squeeze()
    obj = objs.squeeze()

    true_sol_dict = nx.get_edge_attributes(multigraph, "sp")
    true_sol = np.array(list(true_sol_dict.values()))
    true_obj = np.dot(cost, true_sol)

    assert np.isclose(-obj, true_obj), "model and true objective values differ"
    assert (sol == true_sol).all(), "model and true paths differ"
