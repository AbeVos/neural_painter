import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 7, stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(),
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
        # x = self.fc(x)

        mean = self.mean(x)
        logvar = self.logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.linear = nn.Linear(latent_dim, 512)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 5),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 7, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, 7, stride=2, padding=2, output_padding=1)
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(len(z), 512, 1, 1)

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
        # print(D_kl.item(), recon_loss.item())
        elbo = recon_loss + D_kl

        return elbo

    def sample(self, n_samples, z=None):
        if z is None:
            z = torch.randn(n_samples, self.latent_dim).to(self.device)

        images = torch.sigmoid(self.decoder(z)).detach()

        return images, z


def save_elbo_plot(elbo, filename):
    plt.figure()
    plt.plot(elbo)
    plt.xlabel("Step")
    plt.ylabel("ELBO")
    plt.savefig(filename)
    plt.close()


def save_sample_plot(samples, filename, nrow=5):
    plt.figure()
    grid = make_grid(samples, nrow=nrow).cpu()
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


def train(dataset, model, optimizer, device, num_epochs=100,
          batch_size=128, eval_interval=10):
    elbo_plot = []
    z = None

    for epoch in range(num_epochs):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        for idx, batch in enumerate(dataloader):
            model.train()

            images = batch['image']
            images = images.to(device).float()

            elbo = model(images)

            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

            elbo_plot.append(elbo.item())

            if idx % eval_interval == 0:
                model.eval()

                save_elbo_plot(elbo_plot, "elbo.png")

                mean_elbo = sum(elbo_plot[-eval_interval:]) / eval_interval
                print(f"Epoch {epoch:03d}, batch {idx:05d} | "
                      f"ELBO: {mean_elbo}")

                samples, z = model.sample(25, z)
                save_sample_plot(
                    samples, f"samples/samples_{epoch:03d}_{idx:05d}.png")

                torch.save(model.state_dict(), "stroke_vae.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help="")
    parser.add_argument(
        '--batch_size', dest='batch_size', type=int, default=128,
        help="")
    parser.add_argument(
        '--latent_dim', dest='latent_dim', type=int, default=12,
        help="")
    parser.add_argument(
        '--device', dest='device', type=str, default='cuda:0',
        help="")
    parser.add_argument(
        '--eval_interval', dest='eval_interval', type=int, default=10,
        help="")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = BrushStrokeDataset('labels.csv', 'images/', transform=ToTensor())

    model = VAE(args.latent_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train(dataset, model, optimizer, device, args.epochs, args.batch_size,
          args.eval_interval)
