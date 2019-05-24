"""
Train a VAE to learn a latent representation of ImageNet images.
The trained model can be used to compute the content loss of a Neural Painter
when doing intrinsic style transfer.
"""
import os
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from architectures.vae import VAE


def train_epoch(dataset, model, optimizer, eval_interval=10, device='cuda:0',
                sample_path='samples/vae_samples.png',
                model_path='models/content_net.pth',
                plot_path='content_net_loss.png'):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=64,
                            num_workers=4)
    agg_elbo = 0
    elbo_plot = []
    noise = torch.randn(64, model.latent_dim).to(device)

    for idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        elbo = model(images)

        agg_elbo += elbo.item()

        if model.train():
            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

        if idx % eval_interval is 0 and idx > 0:
            mean_elbo = agg_elbo / (eval_interval - 1)
            elbo_plot.append(mean_elbo)
            agg_elbo = 0

            print(f"\tStep: {idx:05d} | ELBO: {mean_elbo}")

            model.eval()
            samples = torch.sigmoid(model.decoder(noise))
            save_image(samples[:64], sample_path, nrow=8)
            torch.save(model.state_dict(), model_path)
            model.train()

            plt.figure()
            plt.plot(elbo_plot)
            plt.xlabel("Step")
            plt.ylabel("ELBO")
            plt.savefig(plot_path)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE on 64x64 ImageNet")
    parser.add_argument(
        '-m', '--model', dest='model', type=str, default='models/content_net.pth',
        help="Path to save the model to. If the `continue` flag is set, "
        "the model at this path will be loaded and overwritten.")
    parser.add_argument(
        '-c', '--continue', dest='continuing', action='store_true',
        help="If set, continue training the model at path `model`.")
    parser.add_argument('--device', dest='device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)

    train_data = ImageFolder('train_64x64/', transform=transforms.ToTensor())
    # valid_data = ImageFolder('valid_64x64/', transform=transforms.ToTensor())

    vae = VAE(latent_dim=1024, device=device)

    if args.continuing:
        vae.load_state_dict(torch.load(args.model))

    epochs = 10

    lr = torch.linspace(5e-5, 1e-5, epochs)

    for epoch in range(epochs):
        print(f"Epoch: {epoch:03d}")

        optimizer = optim.Adam(vae.parameters(), lr=lr[epoch])
        train_epoch(train_data, vae, optimizer, device=device,
                    sample_path=f"samples/content_{epoch:03d}.png",
                    model_path=args.model)
