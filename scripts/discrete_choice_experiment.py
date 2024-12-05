import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
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
from utils.wandb import record_metrics, get_friendly_name

matplotlib.use("agg")


class Encoder(nn.Module):
    def __init__(self, obs_dim, context_dim, hidden_dims, latent_dim):
        super().__init__()
        self.params_per_latent = 2
        self.mlp = MLP(
            obs_dim + context_dim[0] * context_dim[1],
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
        self.beta = nn.Linear(context_dim[1], 1)
        self.mlp = MLP(
            latent_dim,
            hidden_dims + [obs_dim],
            activation_layer=nn.LeakyReLU,
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor, norm: bool = False):
        determ = self.beta(context.flatten(0, 1)).reshape(context.shape[:2])
        random = self.mlp(latents)
        util = determ + random
        return util


class OneHotCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, choices: torch.Tensor, reduction: str = "mean"):
        labels = torch.argmax(choices, dim=-1)
        return nn.functional.cross_entropy(logits, labels, reduction=reduction)


def train_step(model: nn.Module, obs: torch.Tensor, context: torch.Tensor, kld_weight: float):
    kld, reconstruction_loss = model(obs, context, n_samples=10)
    # maximize elbo = minimize -elbo = minimize kld - log likelihood = minimize kld + nll
    loss = kld_weight * kld + (1 - kld_weight) * reconstruction_loss
    metrics = {"kld": kld.detach(), "reconstruction_loss": reconstruction_loss.detach(), "loss": loss.detach()}
    return loss, metrics


def val_step(model: nn.Module, obs: torch.Tensor, context: torch.Tensor):
    utils = model.sample(context)
    return utils


def get_argparser():
    parser = argparse.ArgumentParser(
        "Experiment with unobserved inter- and intra-individual heterogeneity discrete choice problem"
    )

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("--n_people", type=int, default=1000, help="Number of people")
    dataset_args.add_argument("--n_obs_per_person", type=int, default=16, help="Number of observations per person")
    dataset_args.add_argument(
        "--corr_strength",
        type=float,
        default=0.3,
        choices=[0.3, 0.6],
        help="Strength of correlations between alternatives",
    )

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument("--latent_dim", type=int, default=1, help="Latent dimensions")
    model_args.add_argument(
        "--encoder_hidden_dim",
        type=int,
        nargs="*",
        default=[512, 256, 128, 64],
        help="Hidden dimensions in encoder MLP",
    )
    model_args.add_argument(
        "--decoder_hidden_dim",
        type=int,
        nargs="*",
        default=[64, 128, 256, 512],
        help="Hidden dimensions in decoder MLP",
    )

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    train_args.add_argument("--n_epochs", type=int, default=500, help="Maximum number of training epochs")
    train_args.add_argument("--lr", type=float, default=1e-2, help="Optimizer learning rate")
    train_args.add_argument(
        "--kld_weight", type=float, default=1e-2, help="Relative weight of KL divergence and reconstruction loss"
    )

    wandb_args = parser.add_argument_group("wandb", description="W&B arguments")
    wandb_args.add_argument("--use_wandb", action="store_true", help="Upload metrics to W&B")
    wandb_args.add_argument("--wandb_project", type=str, default=pathlib.Path(__file__).stem, help="W&B project")

    return parser


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
    return plot.figure


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
    U = torch.bmm(X, betas.unsqueeze(2)).squeeze()
    # U += Gumbel(0, 1).sample(U.shape)
    return U


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()

    if args.use_wandb:
        run = wandb.init(project=args.wandb_project, config=args, name=get_friendly_name(args, argparser))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_alternatives = 5
    n_feats = 4
    n_samples = args.n_people * args.n_obs_per_person
    data_generator = DiscreteChoice(
        n_alternatives, partial(util_fn, args.n_people, args.n_obs_per_person, args.corr_strength)
    )

    feats = Uniform(0, 2).sample((n_samples, n_alternatives, n_feats))
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

    train_bs, test_bs = min(args.batch_size, len(train_set)), min(args.batch_size, len(test_set))
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, drop_last=True)

    obs_feats = n_feats
    prior = Normal(torch.zeros(args.latent_dim, device=device), torch.ones(args.latent_dim, device=device))
    encoder = Encoder(n_alternatives, (n_alternatives, obs_feats), args.encoder_hidden_dim, args.latent_dim)
    decoder = Decoder(args.latent_dim, (n_alternatives, obs_feats), args.decoder_hidden_dim, n_alternatives)
    model = CVAE(prior, decoder, encoder, OneHotCrossEntropy())
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = defaultdict(Average)

    for epoch in tqdm.trange(args.n_epochs):
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            obs = batch.choices.to(device)
            context = batch.feats.to(device)
            loss, train_metrics = train_step(model, obs, context, args.kld_weight)

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

            utils = torch.cat(utils, dim=0)
            utils_pred = torch.cat(utils_pred, dim=0)

            figure = plot_util_dist(utils.numpy(), utils_pred.numpy())
            if args.use_wandb:
                img = fig_to_rgb_tensor(figure)
                wandb.log({"util": wandb.Image(img)}, step=epoch)
            else:
                figure.savefig(f"util_epoch={epoch}.png")
            plt.close()

        if args.use_wandb:
            record_metrics(metrics, epoch)
