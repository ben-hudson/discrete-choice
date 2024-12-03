import numpy as np
import torch

from typing import Callable


class DiscreteChoice:
    def __init__(self, n_alternatives: int, util_fn: Callable, seed: int = None):
        self.n_alternatives = n_alternatives
        self.get_util = util_fn
        self.seed = seed

    def sample(self, feats: np.ndarray, n_samples: int = 1):
        util = self.get_util(feats, n_samples, self.seed)
        if isinstance(util, torch.Tensor):  # keep everything to np.array for consistency
            util = util.numpy()
        assert len(util.shape) == 2, f"expected util to be a 2D array, but got {len(util.shape)}D"
        assert (util.shape[0] == n_samples) and (
            util.shape[1] == self.n_alternatives
        ), f"expected util to be a {n_samples}x{self.n_alternatives} tensor, but got {util.shape[0]}x{util.shape[1]}"

        choices = np.argmax(util, axis=1, keepdims=True)
        choice_util = np.take_along_axis(util, choices, axis=1)
        other = {}
        return util, choices.squeeze(), choice_util.squeeze(), other
