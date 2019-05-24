import torch
import torch.nn as nn


class Interpolate(nn.Module):
    """
    Module wrapper for nn.functional.interpolate.
    """
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
    """
    Replaces the ConvTranspose2d layer by upsampling the image and
    applying a regular Conv2d layer to it.
    """
    def __init__(self, scale_factor, channels_in, channels_out, kernel_size):
        super(Upsample, self).__init__()

        self.layers = nn.Sequential(
            Interpolate(scale_factor, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(channels_in, channels_out, kernel_size, stride=1,
                      padding=0)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    """
    Encoder based on the DCGAN architecture.
    """
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mean = nn.Linear(1024, latent_dim)
        self.logvar = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(len(x), -1)

        return self.mean(x), self.logvar(x)


class Decoder(nn.Module):
    """
    Decoder based on the DCGAN architecture.
    """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            Upsample(2, 1024, 512, 5),
            nn.LeakyReLU(0.2, inplace=True),

            Upsample(2, 512, 256, 5), 
            nn.LeakyReLU(0.2, inplace=True),

            Upsample(2, 256, 128, 5), 
            nn.LeakyReLU(0.2, inplace=True),

            Upsample(2, 128, 3, 5), 
        )

    def forward(self, z):
        z = z.view(len(z), self.latent_dim, 1, 1)

        return self.layers(z)


class VAE(nn.Module):
    def __init__(self, latent_dim, device, beta=1):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.encoder = Encoder(latent_dim).to(device)
        self.decoder = Decoder(latent_dim).to(device)

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
