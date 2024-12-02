import networkx as nx
import numpy as np

from scipy.optimize import linprog
from typing import Any, Callable

from .simple import DiscreteChoice


class PathChoice(DiscreteChoice):
    def __init__(self, graph: nx.Graph, util_fn: Callable, seed: int = None):
        if not graph.is_directed():
            print("converting to directed graph")
            self.graph = graph.to_directed()
        else:
            self.graph = graph

        self.rng = np.random.default_rng(seed=seed)

        super().__init__(len(self.graph.edges), util_fn, seed)

    def sample(self, feats: np.ndarray, n_samples: int = 1, fixed_orig: Any = None, fixed_dest: Any = None):
        util, _, _ = super().sample(feats, n_samples)

        ods = self._sample_ods(n_samples, fixed_orig, fixed_dest)
        constr_mat, constr_levels = self._ods_to_constrs(ods)
        choices, choice_util = self._solve(util, constr_mat, constr_levels)
        return util, choices, choice_util, constr_mat, constr_levels

    def _sample_ods(self, n_samples: int = 1, fixed_orig: Any = None, fixed_dest: Any = None):
        if fixed_orig is not None and fixed_dest is not None:
            assert nx.has_path(self.graph, fixed_orig, fixed_dest), f"no path between {fixed_orig} and {fixed_dest}"

        ods = []
        while len(ods) < n_samples:
            if fixed_orig is not None:
                orig = fixed_orig
            else:
                orig = self.rng.choice(self.graph.nodes)

            if fixed_dest is not None:
                dest = fixed_dest
            else:
                dest = self.rng.choice(self.graph.nodes)

            if orig != dest and nx.has_path(self.graph, orig, dest):
                ods.append((orig, dest))

        return ods

    def _ods_to_constrs(self, ods):
        node_idx = {n: i for i, n in enumerate(self.graph.nodes)}
        constr_levels = np.zeros((len(ods), len(self.graph.nodes)))
        for constr_level, (orig, dest) in zip(constr_levels, ods):
            constr_level[node_idx[orig]] = -1
            constr_level[node_idx[dest]] = 1

        constr_mat = nx.incidence_matrix(self.graph, oriented=True).toarray()

        return constr_mat, constr_levels

    def _solve(self, utils: np.ndarray, constr_mat: np.ndarray, constr_levels: np.ndarray):
        assert utils.shape[0] == constr_levels.shape[0]
        assert constr_mat.shape[0] == constr_levels.shape[1]
        assert constr_mat.shape[1] == utils.shape[1]

        choices = []
        choice_utils = []
        for u, b in zip(utils, constr_levels):
            result = linprog(-u, A_eq=constr_mat, b_eq=b, bounds=(0, 1), integrality=True)
            assert result.status == 0, f"optimization terminated with non-zero status {result.status}: {result.message}"
            choices.append(result.x)
            choice_utils.append(-result.fun)

        return np.vstack(choices), np.array(choice_utils)
