import torch
import torch.nn as nn

from .vae import VAE


class VAEPainter(VAE):
    def __init__(self, latent_dim, device, beta=1):
        super(VAEPainter, self).__init__(latent_dim, device, 1, beta)

        self.colors = None

        '''
        self.recon_loss = nn.BCELoss(reduction='sum')
        self.encoder = StrokeEncoder(latent_dim).to(device)
        self.decoder = StrokeDecoder(latent_dim, device).to(device)
        '''

    def forward(self, x):
        alpha = x[:, -1].unsqueeze(1)

        return super(VAEPainter, self).forward(alpha)

    def sample(self, n_samples, z):
        # TODO: Sample alpha map and combine with random color.

        if z is None:
            z = torch.randn((n_samples, self.latent_dim)).to(self.device)

        if self.colors is None:
            self.colors = torch.rand((n_samples, 3, 1, 1)).to(self.device)

        alpha = torch.sigmoid(self.decoder(z))
        image = torch.ones((n_samples, 3, 64, 64)).to(self.device)
        image *= self.colors
        image = torch.cat((image, alpha), dim=1)

        return image, z


class ActionEncoder(nn.Module):
    """
    Transform elements from the action space to a VAEPainter's latent space.
    """
    def __init__(self, action_dim, latent_dim, hidden_dim, hidden_layers):
        super(ActionEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.layers(x)
