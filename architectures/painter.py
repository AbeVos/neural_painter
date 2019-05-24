import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, latent_dim, channels_in=4):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 7, stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )

        self.mean = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(len(x), -1)
        x = self.fc(x)

        mean = self.mean(x)
        logvar = self.logvar(x)

        return mean, logvar


class _Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim - 3, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 7, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 7, stride=2, padding=2,
                               output_padding=1),
            nn.Sigmoid()
        )

        self.color = nn.Sequential(
            nn.Linear(3, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, z):
        color = self.color(z[:, -3:])
        z = z[:, :-3]
        z = self.fc(z)
        z = z.view(len(z), 512, 1, 1)

        # return self.upsample(z)
        alpha = self.conv(z)
        alpha = alpha.repeat((1, 4, 1, 1))
        alpha[:, :3, ...] *= color[..., None, None]

        return alpha


class VAEPainter(nn.Module):
    def __init__(self, latent_dim, device, beta=1):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.encoder = _Encoder(latent_dim, channels_in=3).to(device)
        self.decoder = _Decoder(latent_dim, channels_out=3).to(device)

        self.recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.beta = beta

    def forward(self, image):
        def kl_divergence(mean, logvar):
            return 0.5 * torch.sum(logvar.exp() + mean.pow(2) - 1 - logvar, -1)

        mean, logvar = self.encoder(image)

        # Sample a latent vector.
        noise = torch.randn(mean.shape).to(self.device)
        z = noise * torch.exp(0.5 * logvar) + mean

        recon = self.decoder(z)

        D_kl = kl_divergence(mean, logvar).mean()
        recon_loss = self.recon_loss(recon, image) / len(image)
        elbo = recon_loss + self.beta * D_kl

        return elbo

    def sample(self, n_samples, z=None):
        if z is None:
            z = torch.randn(n_samples, self.latent_dim).to(self.device)

        images = self.decoder(z).detach()

        return images, z
