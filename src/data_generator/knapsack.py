import networkx as nx
import numpy as np

from scipy.optimize import linprog
from typing import Any, Callable

from .simple import DiscreteChoice


class LinAddConstrBundleChoice(DiscreteChoice):
    def sample(
        self,
        feats: np.ndarray,
        constr_mat: np.ndarray,
        constr_levels: np.ndarray,
        n_samples: int = 1,
        eq_constr: bool = False,
    ):
        util, _, _, _ = super().sample(feats, n_samples)
        choices, choice_util = self._solve(util, constr_mat, constr_levels, eq_constr)
        return util, choices, choice_util, {}

    def _solve(self, utils: np.ndarray, constr_mat: np.ndarray, constr_levels: np.ndarray, eq_constr: bool):
        assert utils.shape[0] == constr_levels.shape[0]
        assert constr_mat.shape[0] == constr_levels.shape[1]
        assert constr_mat.shape[1] == utils.shape[1]

        choices = []
        choice_util = []
        for u, b in zip(utils, constr_levels):
            if eq_constr:
                result = linprog(-u, A_eq=constr_mat, b_eq=b, bounds=(0, 1), integrality=True)
            else:
                result = linprog(-u, A_ub=constr_mat, b_ub=b, bounds=(0, 1), integrality=True)
            assert result.status == 0, f"optimization terminated with non-zero status {result.status}: {result.message}"
            choices.append(result.x)
            choice_util.append(-result.fun)

        return np.vstack(choices), np.array(choice_util)
