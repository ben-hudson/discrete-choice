import gurobipy as gp
import networkx as nx
import torch

from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel
from typing import Tuple
from utils.misc import hush


# an ILP formulation of the SP problem
# unfortunately, we have to stick to the PyEPO formula so it is compatible with other parts of the package
# all __init__ args MUST BE CLASS MEMBERS WITH THE SAME NAME because of the stupid way the model is called in the loss functions
class ILPShortestPath(optGrbModel):
    def __init__(self, graph: nx.DiGraph, source, sink) -> None:
        self.graph = graph
        self.source = source
        self.sink = sink

        assert self.graph.is_directed(), f"the graph should be directed"
        assert self.source in self.graph.nodes, f"source node {self.source} does not exist in the graph"
        assert self.sink in self.graph.nodes, f"sink node {self.sink} does not exist in the graph"

        super().__init__()  # sets self._model and self.x via _getModel()

    def setObj(self, costs: torch.Tensor) -> None:
        assert len(self.x) == len(costs)
        obj = gp.quicksum(c * self.x[key] for c, key in zip(costs, self.x))
        self._model.setObjective(obj)

    def solve(self) -> Tuple[torch.FloatTensor, float]:
        with hush():
            sol, obj = super().solve()
        if self._model.Status != GRB.OPTIMAL:
            raise Exception(f"Solve failed with status code {self._model.Status}")

        return torch.FloatTensor(sol), obj

    def _getModel(self) -> Tuple[gp.Model, gp.Var]:
        model = gp.Model("SP")
        model.ModelSense = GRB.MINIMIZE

        # returns a dict-like object where the edges are keys
        x = model.addVars(self.graph.edges, name="x", vtype=GRB.BINARY)

        # flow conservation on each node
        for n in self.graph.nodes:
            expr = 0

            # annoying, but this is how networkx works
            if self.graph.is_multigraph():
                for e in self.graph.in_edges(n, keys=True):
                    expr += x[e]
                for e in self.graph.out_edges(n, keys=True):
                    expr -= x[e]

            else:
                for e in self.graph.in_edges(n):
                    expr += x[e]
                for e in self.graph.out_edges(n):
                    expr -= x[e]

            if n == self.source:
                model.addConstr(expr == -1)
            elif n == self.sink:
                model.addConstr(expr == 1)
            else:
                model.addConstr(expr == 0)

        return model, x
