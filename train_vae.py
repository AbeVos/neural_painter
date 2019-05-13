import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models import VAE
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


def save_elbo_plot(elbo, filename):
    """
    Save an image of the ELBO plot.
    """
    plt.figure()
    plt.plot(elbo)
    plt.xlabel("Step")
    plt.ylabel("ELBO")
    plt.savefig(filename)
    plt.close()


def save_sample_plot(samples, filename, nrow=5):
    """
    Save a plot of a number of images sampled from the generator.
    """
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

                save_elbo_plot(elbo_plot, "elbo.png")

                mean_elbo = sum(elbo_plot[-eval_interval:]) / eval_interval
                print(f"Epoch {epoch:03d}, batch {idx:05d} | "
                      f"ELBO: {mean_elbo}")

        model.eval()
        samples, z = model.sample(25, z)
        os.makedirs("samples/", exist_ok=True)
        save_sample_plot(
            samples, f"samples/samples_{epoch+1:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help="Number of epochs to train for.")
    parser.add_argument(
        '--batch_size', dest='batch_size', type=int, default=128,
        help="Batch size for training.")
    parser.add_argument(
        '--latent_dim', dest='latent_dim', type=int, default=12,
        help="Size of the latent dimension.")
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

    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/stroke_vae.pth")
