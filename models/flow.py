import math
import torch
import torch.nn as nn

from .autoencoder import Encoder, Decoder
from .util import split_mask


class ConditionalPrior(nn.Module):
    def __init__(self, dim_z, dim_x, dim_hidden=128, device='cpu'):
        """
        A Gaussian conditional prior whose parameters are modelled by
        a neural network.

        Parameters
        ----------
        dim_x: int
            Dimensionality of the conditional.
        dim_z: int
            Dimensionality of the prior.
        dim_hidden: int, optional, default 64
            Dimensionality of the hidden state.
        """
        super(ConditionalPrior, self).__init__()

        assert dim_x > 0
        assert dim_z > 0

        self.dim_z = dim_z

        self.conditional_layers = nn.Sequential(
            nn.Linear(dim_x, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, 2 * dim_z),
        ).to(device)

        # Constant value to compute Gaussian likelihood.
        self.gaussian_normalizer = torch.Tensor(
            [math.log(2 * math.pi)]).to(device)

        self.device = device

    def forward(self, x):
        mu, logvar = self.conditional_layers(x).split(self.dim_z, dim=-1)

        return mu, logvar

    def sample(self, x):
        mean, logvar = self(x)

        z = torch.randn_like(mean).to(self.device)

        return mean + z * torch.exp(0.5 * logvar)

    def log_prob(self, z, condition):
        """
        Compute the prior's conditional likelihood.
        """
        mean, logvar = self(condition)

        z = (z - mean) / logvar.exp().sqrt()

        log_prob = -0.5 * (
            self.dim_z * self.gaussian_normalizer
            + (z.unsqueeze(1) @ z.unsqueeze(-1)).squeeze()
            + logvar.sum(-1)
        )

        return log_prob


class Permutation(nn.Module):
    """
    A random permutation.
    """
    def __init__(self, n_features, device='cpu'):
        super(Permutation, self).__init__()

        indices = torch.randperm(n_features).to(device)
        self.indices = nn.Parameter(indices, requires_grad=False)
        self.inverse_indices = nn.Parameter(n_features - 1 - indices,
                                            requires_grad=False)

    def forward(self, x):
        return x[:, self.indices]

    def inverse(self, x):
        return x[:, self.inverse_indices]


class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim_feature, dim_condition, mask, dim_hidden=64):
        super(ConditionalCouplingLayer, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_feature + dim_condition, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, 2 * dim_feature),
        )

        self.mask = mask

        self.dim_feature = dim_feature

    def forward(self, x, condition):
        x_a = self.mask * x

        transform = self.network(torch.cat([x_a, condition], dim=-1))
        scale, translate = transform.split(self.dim_feature, dim=-1)
        scale = torch.tanh(scale)

        y = x_a + (1 - self.mask) * (x * torch.exp(scale) + translate)

        log_det_J = scale.mul(1 - self.mask)\
            .view(len(x), -1)\
            .sum(1, keepdim=True)

        return y, log_det_J

    def inverse(self, y, condition):
        y_a = self.mask * y

        transform = self.network(torch.cat([y_a, condition], dim=-1))
        scale, translate = transform.split(self.dim_feature, dim=-1)
        scale = torch.tanh(scale)

        x = y_a + (1 - self.mask) * (y - translate) * torch.exp(-scale)

        return x


class ConditionalCouplingBlock(nn.Module):
    """
    Combine two conditional coupling layers and a random permutation layer
    to create a conditional coupling block.
    """
    def __init__(self, dim_feature, dim_condition, device='cpu'):
        super(ConditionalCouplingBlock, self).__init__()

        mask = split_mask(dim_feature).to(device)

        self.affine_1 = ConditionalCouplingLayer(
            dim_feature, dim_condition, mask
        ).to(device)
        self.affine_2 = ConditionalCouplingLayer(
            dim_feature, dim_condition, 1 - mask
        ).to(device)
        self.permutation = Permutation(dim_feature, device)

    def forward(self, x, condition):
        x, jacobian_1 = self.affine_1(x, condition)
        x, jacobian_2 = self.affine_2(x, condition)
        x = self.permutation(x)

        return x, jacobian_1 + jacobian_2

    def inverse(self, x, condition):
        x = self.permutation.inverse(x)
        x = self.affine_2.inverse(x, condition)
        x = self.affine_1.inverse(x, condition)

        return x


class ConditionalFlow(nn.Module):
    """
    Combine a number of conditional coupling blocks into a conditional
    normalizing flow.
    """
    def __init__(self, dim_feature, dim_condition, n_layers, device='cpu'):
        super(ConditionalFlow, self).__init__()

        self.layers = nn.ModuleList([
            ConditionalCouplingBlock(dim_feature, dim_condition, device)
            for _ in range(n_layers)
        ])

    def forward(self, x, condition):
        log_det_J = 0

        for layer in self.layers:
            x, J = layer(x, condition)

            log_det_J += J

        return x, log_det_J

    def inverse(self, x, condition):
        for layer in reversed(self.layers):
            x = layer.inverse(x, condition)

        return x


class GenerativeLatentFlow(nn.Module):
    """
    An autoencoder whose latent space is modelled using a Normalizing Flow.
    """
    def __init__(self, dim_latent, dim_condition, device='cpu'):
        super(GenerativeLatentFlow, self).__init__()

        self.encoder = Encoder(dim_latent).to(device)
        self.decoder = Decoder(dim_latent).to(device)

        self.prior = ConditionalPrior(dim_latent, dim_condition, device=device)
        self.flow = ConditionalFlow(dim_latent, dim_condition, 10, device)

    def encode(self, x, condition):
        latent = self.encoder(x)
        z, _ = self.flow(latent, condition)

        return z

    def decode(self, z, condition):
        latent = self.flow.inverse(z, condition)
        x = self.decoder(latent)

        return x

    def forward(self, x, condition):
        latent = self.encoder(x)
        z, log_det_J = self.flow(latent.data, condition)
        latent_recon = self.flow.inverse(z, condition)
        x_recon = self.decoder(latent_recon.data)
        # x_recon = self.decoder(latent)

        return x_recon, z, log_det_J
    
    def sample(self, condition):
        z = self.prior.sample(condition)

        return self.decode(z, condition)
