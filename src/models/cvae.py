import torch
import torch.distributions as D

from typing import Tuple

from .vae import VAE


class CVAE(VAE):
    def forward(
        self, obs: torch.Tensor, context: torch.Tensor, n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior = self.recognition_model(obs, context)

        kld = D.kl_divergence(posterior, self.prior).mean()

        latent_sample = posterior.rsample((n_samples,))
        losses = []
        for i in range(n_samples):
            obs_hat = self.generation_model(latent_sample[i], context)
            losses.append(self.reconstruction_loss(obs_hat, obs, reduction="none"))
        loss = torch.stack(losses).mean()

        return kld, loss

    def sample(self, context: torch.Tensor) -> torch.Tensor:
        latent_sample = self.prior.sample((context.size(0),))
        obs_hat = self.generation_model(latent_sample, context)
        return obs_hat
