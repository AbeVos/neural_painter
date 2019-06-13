import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from architectures.painters import VAEPainter
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


def save_sample_plot(samples, filename, nrow=8):
    """
    Save a plot of a number of images sampled from the generator.
    """
    samples = samples.detach().cpu()
    color = samples[:, :3]
    alpha = samples[:, -1].unsqueeze(1)
    background = torch.ones((len(samples), 3, 64, 64))
    background[32:] *= 0

    images = alpha * color + (1 - alpha) * background
    save_image(images, filename, nrow=nrow)


def train(dataset, model, optimizer, device, num_epochs=100,
          batch_size=128, eval_interval=10):
    elbo_plot = []
    z = None

    for epoch in range(num_epochs):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        elbo_agg = 0

        for idx, batch in enumerate(dataloader):
            model.train()

            images = batch['image']
            images = images.to(device).float()

            elbo = model(images)

            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

            elbo_agg += elbo.item()

            if idx % eval_interval == 0 and idx > 0:

                mean_elbo = elbo_agg / eval_interval
                elbo_agg = 0
                elbo_plot.append(mean_elbo)

                print(f"Epoch {epoch+1:03d}, batch {idx:05d} | "
                      f"ELBO: {mean_elbo}")

                save_elbo_plot(elbo_plot, "elbo.png")

                model.eval()
                samples, z = model.sample(64, z)
                save_sample_plot(samples, f"samples/samples_vae_painter.png")

                if mean_elbo < 1e3:
                    torch.save(model.state_dict(), f"models/{args.model}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=10,
                        help="Number of epochs to train for.")
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=64, help="Batch size for training.")
    parser.add_argument('--latent_dim', dest='latent_dim', type=int, default=8,
                        help="Size of the latent dimension.")
    parser.add_argument('--beta', dest='beta', type=float, default=1,
                        help="Beta-VAE parameter.")
    parser.add_argument('--device', dest='device', type=str, default='cuda:0',
                        help="")
    parser.add_argument('--data_root', dest='root', type=str,
                        default='images_100k/', help="")
    parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=50,
                        help="")
    parser.add_argument('--model', dest='model', type=str, default='painter.pth',
                        help="Name of the saved model.")
    args = parser.parse_args()

    os.makedirs("samples/", exist_ok=True)
    device = torch.device(args.device)

    dataset = BrushStrokeDataset(os.path.join(args.root, 'labels.csv'),
                                 os.path.join(args.root, 'images/'),
                                 transform=ToTensor())

    model = VAEPainter(args.latent_dim, device, beta=args.beta)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("models/", exist_ok=True)
    train(dataset, model, optimizer, device, args.epochs, args.batch_size,
          args.eval_interval)
