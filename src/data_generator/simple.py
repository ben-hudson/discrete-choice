import torch

from typing import Tuple, Callable


class DiscreteChoice:
    def __init__(
        self,
        n_alternatives: int,
        taste_dist: torch.distributions.Distribution,
        util_func: Callable,
    ):
        self.n_alternatives = n_alternatives
        self.taste_dist = taste_dist
        self.get_util = util_func

    def get_choices(
        self, obs_vars: torch.Tensor, unobs_vars: torch.Tensor, n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tastes = self.taste_dist.sample((n_samples,))

        util = self.get_util(obs_vars, unobs_vars, tastes).squeeze()
        assert util.dim() == 2, f"expected util to be a 2D tensor after squeezing, but got {util.dim()}D"
        assert (util.size(0) == n_samples) and (
            util.size(1) == self.n_alternatives
        ), f"expected util to be a {n_samples}x{self.n_alternatives} tensor, but got {util.size(0)}x{util.size(1)}"

        choice_util, choices = torch.max(util, dim=-1)
        # feats, costs, sols, objs
        return obs_vars, util, choices, choice_util
