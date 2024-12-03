import numpy as np

from data_generator.knapsack import LinAddConstrBundleChoice


def test_knapsack_choice():
    n_samples = 10
    n_alternatives = 5

    def util(costs, n_samples, seed):
        utils = costs.copy()
        np.random.shuffle(utils)
        return np.broadcast_to(utils, (n_samples, *utils.shape))

    choice_model = LinAddConstrBundleChoice(n_alternatives, util)

    prices = np.arange(n_alternatives)
    budgets = np.arange(n_samples).reshape(-1, 1)

    util, choices, _, _ = choice_model.sample(prices, prices.reshape(1, -1), budgets, len(budgets))

    used_budgets = choices @ prices.reshape(-1, 1)
    assert (used_budgets <= budgets).all(), "bundle exceeds budget"
