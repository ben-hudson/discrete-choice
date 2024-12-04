import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
import wandb

from collections import defaultdict
from datasets import DiscreteChoice, ChoiceDataset
from functools import partial
from ignite.metrics import Average
from models.cvae import CVAE
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.distributions import Normal, MultivariateNormal, Gumbel, Uniform
from torch.utils.data import DataLoader
from torchvision.ops import MLP
from utils.misc import fig_to_rgb_tensor
from utils.wandb import record_metrics


class Encoder(nn.Module):
    def __init__(self, obs_dim, context_dim, hidden_dims, latent_dim):
        super().__init__()
        self.params_per_latent = 2
        self.mlp = MLP(
            obs_dim + context_dim,
            hidden_dims + [latent_dim * self.params_per_latent],
            activation_layer=nn.LeakyReLU,
        )

    def forward(self, obs: torch.Tensor, context: torch.Tensor):
        x = torch.cat([obs, context.flatten(1)], dim=-1)
        params = self.mlp(x)
        mean, logvar = params.chunk(self.params_per_latent, dim=-1)
        var = logvar.exp() + 1e-6
        return Normal(mean, var.sqrt())


class Decoder(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dims, obs_dim):
        super().__init__()
        self.mlp = MLP(
            latent_dim + context_dim,
            hidden_dims + [obs_dim],
            activation_layer=nn.LeakyReLU,
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor, norm: bool = False):
        x = torch.cat([latents, context.flatten(1)], dim=-1)
        logits = self.mlp(x)
        if norm:
            return nn.functional.softmax(logits)
        else:
            return logits


class OneHotCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, choices: torch.Tensor, reduction: str = "mean"):
        labels = torch.argmax(choices, dim=-1)
        return nn.functional.cross_entropy(logits, labels, reduction=reduction)


def train_step(model: nn.Module, obs: torch.Tensor, context: torch.Tensor, kld_weight: float):
    kld, reconstruction_loss = model(obs, context)
    # maximize elbo = minimize -elbo = minimize kld - log likelihood = minimize kld + nll
    loss = kld_weight * kld + (1 - kld_weight) * reconstruction_loss
    metrics = {"kld": kld.detach(), "reconstruction_loss": reconstruction_loss.detach(), "loss": loss.detach()}
    return loss, metrics


def val_step(model: nn.Module, obs: torch.Tensor, context: torch.Tensor):
    logits = model.sample(context)
    utils = logits.exp()
    return utils


def plot_util_dist(util, util_pred):
    util_normed = StandardScaler().fit_transform(util.reshape(-1, 1)).reshape(util.shape)
    util_df = pd.DataFrame(util_normed)
    util_df["type"] = "true"

    util_pred_normed = StandardScaler().fit_transform(util_pred.reshape(-1, 1)).reshape(util_pred.shape)
    util_pred_df = pd.DataFrame(util_pred_normed)
    util_pred_df["type"] = "pred"

    df = pd.concat((util_df, util_pred_df))
    plot = sns.pairplot(
        df,
        hue="type",
        kind="hist",
        plot_kws={"bins": 10, "alpha": 0.8},
        diag_kws={"bins": 10},
        height=1,
        aspect=1,
        grid_kws={"layout_pad": 0.0},
    )
    return fig_to_rgb_tensor(plot.figure)


# utility function from https://arxiv.org/pdf/1905.00419
def util_fn(N, T, alpha, X, n_samples, seed):
    assert N * T == n_samples, f"expected N*T==n_samples, but got {N}*{T}!={n_samples}"

    zeta = torch.FloatTensor([-0.5, 0.5, -0.5, 0.5])
    sigma_B = (2 * 2 / 3 * zeta.abs()).sqrt()
    sigma_W = (2 * 1 / 3 * zeta.abs()).sqrt()
    Omega_B = torch.eye(4) + alpha * torch.FloatTensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    Omega_W = torch.eye(4) + alpha * torch.FloatTensor([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]])
    Sigma_B = sigma_B.diag() @ Omega_B @ sigma_B.diag()
    Sigma_W = sigma_W.diag() @ Omega_W @ sigma_W.diag()

    mus = MultivariateNormal(zeta, Sigma_B).sample((N,))
    betas = torch.cat([MultivariateNormal(mu, Sigma_W).sample((T,)) for mu in mus])
    U = torch.bmm(betas.unsqueeze(1), X).squeeze()
    # U += Gumbel(0, 1).sample(U.shape)
    return U


if __name__ == "__main__":
    run = wandb.init(project="discrete_choice_experiment")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_alternatives = 5
    n_feats = 4
    n_individuals = 1000
    n_timesteps = 16
    corr_strength = 0.3
    n_samples = n_individuals * n_timesteps
    data_generator = DiscreteChoice(n_alternatives, partial(util_fn, n_individuals, n_timesteps, corr_strength))

    feats = Uniform(0, 2).sample((n_samples, n_feats, n_alternatives))
    util, choices, choice_util, other = data_generator.sample(feats, n_samples)

    feats_normed = StandardScaler().fit_transform(feats.reshape(-1, 1)).reshape(feats.shape).astype(np.float32)
    labels = np.argmax(choices, axis=1)

    n_test = int(0.2 * n_samples)
    n_train = n_samples - n_test
    train_set = ChoiceDataset(
        feats_normed[:n_train], util[:n_train], choices[:n_train], choice_util[:n_train], labels=labels[:n_train]
    )
    test_set = ChoiceDataset(
        feats_normed[n_train:], util[n_train:], choices[n_train:], choice_util[n_train:], labels=labels[n_train:]
    )

    batch_size = 1000
    train_bs, test_bs = min(batch_size, len(train_set)), n_test
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, drop_last=True)

    latent_dim = 4
    obs_feats = n_feats
    prior = Normal(torch.zeros(latent_dim, device=device), torch.ones(latent_dim, device=device))
    encoder = Encoder(n_alternatives, obs_feats * n_alternatives, [512, 256, 128, 64], latent_dim)
    decoder = Decoder(latent_dim, obs_feats * n_alternatives, [64, 128, 256, 512], n_alternatives)
    model = CVAE(prior, decoder, encoder, OneHotCrossEntropy())
    model.to(device)

    lr = 1e-3
    kld_weight = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics = defaultdict(Average)

    n_epochs = 100
    for epoch in tqdm.trange(n_epochs):
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            obs = batch.choices.to(device)
            context = batch.feats.to(device)
            loss, train_metrics = train_step(model, obs, context, kld_weight)

            loss.backward()
            optimizer.step()

            for name, value in train_metrics.items():
                metrics["train/" + name].update(value)

        with torch.no_grad():
            utils = []
            utils_pred = []
            for batch in test_loader:
                obs = batch.choices.to(device)
                context = batch.feats.to(device)
                util_pred = val_step(model, obs, context)
                utils.append(batch.util)
                utils_pred.append(util_pred.cpu())

            utils = torch.cat(utils, dim=-1)
            utils_pred = torch.cat(utils_pred, dim=-1)

            img = plot_util_dist(utils.numpy(), util_pred.numpy())
            wandb.log({"util": wandb.Image(img)}, step=epoch)

        record_metrics(metrics, epoch)
