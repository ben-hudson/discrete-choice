import networkx as nx
import numpy as np


# the example from https://arxiv.org/abs/1905.00883v1
# see Figure 3
def get_graph():
    G = nx.MultiDiGraph()
    G.add_node("o", pos=(0, 0))
    G.add_node("A", pos=(1, 0))
    G.add_node("B", pos=(2, 0))
    G.add_node("C", pos=(3, 0))
    G.add_node("D", pos=(4, 0))
    G.add_node("E", pos=(0, 1))
    G.add_node("F", pos=(1, 1))
    G.add_node("H", pos=(2, 1))
    G.add_node("I", pos=(3, 1))
    G.add_node("G", pos=(1, 2))
    G.add_node("d", pos=(4, 2))

    G.add_edge("o", "A", travel_time=0.3)
    G.add_edge("A", "B", travel_time=0.1)
    G.add_edge("B", "C", travel_time=0.1)
    G.add_edge("C", "D", travel_time=0.3)
    G.add_edge("o", "E", travel_time=0.4)
    G.add_edge("A", "F", travel_time=0.1)
    G.add_edge("B", "H", travel_time=0.2)
    G.add_edge("C", "I", travel_time=0.1)
    G.add_edge("C", "d", travel_time=0.9)
    G.add_edge("D", "d", travel_time=2.6)
    G.add_edge("E", "G", travel_time=0.3)
    G.add_edge("F", "G", travel_time=0.3)
    G.add_edge("F", "H", travel_time=0.2)
    G.add_edge("H", "d", travel_time=0.5)
    G.add_edge("H", "I", travel_time=0.2)
    G.add_edge("I", "d", travel_time=0.3)
    G.add_edge("G", "H", travel_time=0.6)
    G.add_edge("G", "d", travel_time=0.7)
    G.add_edge("G", "d", travel_time=2.8)

    return G


def util_fn(travel_times, n_samples, seed, noise_scale=1.0):
    travel_time_util = -2.0
    constant_util = -0.01

    determ_util = travel_time_util * travel_times + constant_util * 1
    util = np.broadcast_to(determ_util, (n_samples, *determ_util.shape)).copy()
    util += noise_scale * np.random.randn(*util.shape)
    return util
