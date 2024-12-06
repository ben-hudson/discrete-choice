from typing import Callable
import networkx as nx
import numpy as np
import torch

from models.shortestpath import ILPShortestPath


def test_ilp_sp(random_graph: nx.DiGraph, path_to_sol: Callable):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]

    true_obj, true_path = nx.single_source_bellman_ford(random_graph, source, sink, "cost")
    true_sol = path_to_sol(true_path, random_graph.edges)
    true_sol = torch.FloatTensor(true_sol)

    cost_dict = nx.get_edge_attributes(random_graph, "cost")
    cost = torch.FloatTensor(list(cost_dict.values()))

    model = ILPShortestPath(random_graph, source, sink)
    model.setObj(cost)
    sol, obj = model.solve()

    assert np.isclose(obj, true_obj), "model and true objective values differ"
    assert (sol == true_sol).all(), "model and true paths differ"


def test_multigraph_ilp_sp(multigraph):
    node_list = list(multigraph.nodes)
    source = node_list[0]
    sink = node_list[-1]

    cost_dict = nx.get_edge_attributes(multigraph, "cost")
    cost = torch.FloatTensor(list(cost_dict.values()))

    true_sol_dict = nx.get_edge_attributes(multigraph, "sp")
    true_sol = torch.FloatTensor(list(true_sol_dict.values()))
    true_obj = torch.dot(cost, true_sol)

    model = ILPShortestPath(multigraph, source, sink)
    model.setObj(cost)
    sol, obj = model.solve()

    assert np.isclose(obj, true_obj), "model and true objective values differ"
    assert (sol == true_sol).all(), "model and true paths differ"
