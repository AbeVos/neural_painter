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
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
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


class Decoder(nn.Module):
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


class VAE(nn.Module):
    def __init__(self, latent_dim, device, beta=1):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.encoder = Encoder(latent_dim).to(device)
        self.decoder = Decoder(latent_dim).to(device)

        self.recon_loss = nn.BCELoss(reduction='sum')

        self.beta = beta

    def forward(self, image):
        def kl_divergence(mean, logvar):
            return 0.5 * torch.sum(logvar.exp() + mean.pow(2) - 1 - logvar, -1)

        mean, logvar = self.encoder(image)

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


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim, hidden_layers):
        super(ActionEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(action_dim - 3, hidden_dim),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, latent_dim - 3),
        )

        self.color = nn.Sequential(
            nn.Linear(3, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        color = x[:, 6:9]
        x = torch.cat((x[:, :6], x[:, 9:]), 1)

        x = self.layers(x)
        color = self.color(color)

        return torch.cat((x, color), 1)


class StrokeGenerator(nn.Module):
    def __init__(self, action_dim):
        super(StrokeGenerator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(action_dim, 512, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 512, 5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 5),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 7, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 4, 7, stride=2, padding=2,
                               output_padding=1),
            nn.Tanh(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class StrokeDiscriminator(nn.Module):
    def __init__(self, action_dim):
        super(StrokeDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 7, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.condition = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(inplace=True)
        )

    def forward(self, x, c):
        x = self.conv(x)
        c = self.condition(c)

        x = torch.stack((x.squeeze(), c), -1)

        return self.output(x)


class ActionGenerator(nn.Module):
    def __init__(self, action_dim, timesteps):
        super(ActionGenerator, self).__init__()

        self.timesteps = timesteps

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 7, stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lstm = nn.LSTM(512, 512, 2, dropout=0.0)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = len(x)

        x = self.conv(x)
        x = x.view(batch_size, -1)

        x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)

        x, _ = self.lstm(x)
        x = x.view(self.timesteps * batch_size, -1)
        x = self.fc(x)

        return x
