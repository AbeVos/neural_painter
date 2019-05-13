import torch
import torch.nn as nn


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode,
                        align_corners=self.align_corners)

        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor, channels_in, channels_out, kernel_size):
        super(Upsample, self).__init__()

        self.layers = nn.Sequential(
            Interpolate(scale_factor, mode='bilinear'),
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(channels_in, channels_out, kernel_size)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 7, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 5, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 5, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 5, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU()
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


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 5, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 5, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 7, stride=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 7, stride=2, padding=2, output_padding=1,
                               bias=False)
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(len(z), 512, 1, 1)

        # return self.upsample(z)
        return self.conv(z)


class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.encoder = Encoder(latent_dim).to(device)
        self.decoder = Decoder(latent_dim).to(device)

        self.recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, image):
        def kl_divergence(mean, logvar):
            return 0.5 * torch.sum(logvar.exp() + mean.pow(2) - 1 - logvar, -1)

        mean, logvar = self.encoder(image)

        noise = torch.randn(mean.shape).to(self.device)
        z = noise * torch.exp(0.5 * logvar) + mean

        recon = self.decoder(z)

        D_kl = kl_divergence(mean, logvar).mean()
        recon_loss = self.recon_loss(recon, image) / len(image)
        elbo = recon_loss + D_kl

        return elbo

    def sample(self, n_samples, z=None):
        if z is None:
            z = torch.randn(n_samples, self.latent_dim).to(self.device)

        images = torch.sigmoid(self.decoder(z)).detach()

        return images, z


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim, hidden_layers):
        super(ActionEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.layers(x)
