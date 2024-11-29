import torch

from models.parallel_solver import ParallelSolver

from .simple import DiscreteChoice


class LinAddConstrAssortmentChoice(DiscreteChoice):
    def __init__(
        self,
        n_alternatives: int,
        taste_dist: torch.distributions.Distribution,
        util_func: callable,
        solver: ParallelSolver,
    ):
        super().__init__(n_alternatives, taste_dist, util_func)
        self.solver = solver

    def get_choices(self, obs_vars: torch.Tensor, unobs_vars: torch.Tensor, n_samples: int = 1):
        obs_vars, util, _, _ = super().get_choices(obs_vars, unobs_vars, n_samples)

        choices = self.solver(-util)
        choice_util = util @ choices
        return obs_vars, util, choices, choice_util
