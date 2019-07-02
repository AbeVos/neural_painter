import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor
from architectures.gan import Generator, Discriminator


def save_samples(samples, path, color=None):
    if samples.shape[1] == 4:
        samples = samples[:, :3] * samples[:, -1].unsqueeze(1)

    save_image(samples, path, nrow=8)


def iter_epoch(dataset, G, D, optim_G, optim_D, device, epochs, batch_size,
               eval_interval):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                            drop_last=True)

    criterion = nn.BCEWithLogitsLoss()
    recon = nn.MSELoss()

    ones = torch.ones((batch_size, 1)).to(device)
    zeros = torch.zeros((batch_size, 1)).to(device)

    action_sample = torch.rand(64, dataset.action_dim).to(device)

    for idx, data in enumerate(dataloader):
        images, actions = data['image'], data['action']
        images = images[:, -1].unsqueeze(1)
        images = 2 * images - 1
        images = images.to(device).float()

        actions = actions.to(device).float()

        # Train discriminator.
        G.eval()
        D.train()

        action_fake = torch.rand(len(actions), dataset.action_dim).to(device)
        fake = G(action_fake).detach()
        pred_fake = D(fake, action_fake)
        pred_real = D(images, actions)

        loss_D = criterion(pred_real, ones) \
            + criterion(pred_fake, zeros)

        loss_D.backward()
        optim_D.step()
        D.zero_grad()
        G.zero_grad()

        # Train generator.
        G.train()
        D.eval()

        fake = G(actions)
        pred_fake = D(fake, actions)

        loss_G = criterion(pred_fake, ones) \
            + recon(fake, images)

        loss_G.backward()
        optim_G.step()
        D.zero_grad()
        G.zero_grad()

        if idx % 10 == 0:
            print(f"D: {loss_D.item()}, G: {loss_G.item()}")

            G.eval()

            sample = G(action_sample)
            sample = (sample + 1) / 2
            save_samples(sample, "samples/samples_gan_painter.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', dest='data_root', type=str, default='calligraphy_100k',
        help="")
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help="Number of epochs to train for.")
    parser.add_argument(
        '--batch_size', dest='batch_size', type=int, default=128,
        help="Batch size for training.")
    parser.add_argument(
        '--latent_dim', dest='latent_dim', type=int, default=32,
        help="Size of the latent dimension.")
    parser.add_argument(
        '--device', dest='device', type=str, default='cuda:0',
        help="")
    parser.add_argument(
        '--eval_interval', dest='eval_interval', type=int, default=10,
        help="")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = BrushStrokeDataset('labels.csv', args.data_root,
                                 transform=ToTensor())

    generator = Generator(dataset.action_dim).to(device)
    discriminator = Discriminator(dataset.action_dim).to(device)

    optim_G = optim.Adam(generator.parameters(), lr=1e-4)
    optim_D = optim.Adam(discriminator.parameters(), lr=1e-5)

    for epoch in range(10):
        iter_epoch(dataset, generator, discriminator, optim_G, optim_D,
                   device, args.epochs, args.batch_size,
                   args.eval_interval)

    os.makedirs("models/", exist_ok=True)
    torch.save(generator.state_dict(), "models/calligraphy_gan.pth")
