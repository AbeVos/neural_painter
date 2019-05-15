import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from models import VAE, ActionEncoder, ActionGenerator


def train(model, encoder, decoder, dataset, optimizer, timesteps=20):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32,
                            num_workers=4)

    criterion = nn.BCELoss()

    loss_plot = []

    for idx, [images, _] in enumerate(dataloader):
        while True:
            images = images.to(args.device)
            code = model(images)

            z = encoder(code)
            strokes = decoder(z).view(timesteps, len(images), 3, 64, 64)
            strokes = torch.sigmoid(strokes)

            # Blend the strokes into a single painting.
            paintings = strokes.mean(0)

            loss = criterion(paintings, images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_plot.append(loss.item())

            if idx % 50 == 0:
                results = torch.cat((paintings[:5], images[:5]), 0)

                print(idx, loss.item())
                save_image(results, f"samples/painting_{idx:05d}.png", nrow=5)

                plt.figure()
                plt.plot(loss_plot)
                plt.savefig("action_generator_loss.png")
                plt.close()

            idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--encoder', dest='encoder', type=str,
        help="Pretrained ActionEncoder model.")
    parser.add_argument(
        '--vae', dest='vae', type=str,
        help="Pretrained VAE model.")
    parser.add_argument(
        '--latent_dim', dest='latent_dim', type=int, default=32,
        help="")
    parser.add_argument(
        '--action_dim', dest='action_dim', type=int, default=11,
        help="")
    parser.add_argument(
        '--timesteps', dest='timesteps', type=int, default=20,
        help="")
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help="")
    parser.add_argument(
        '--device', dest='device', type=str, default='cuda:0',
        help="")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = ImageFolder('valid_64x64/', transform=transforms.ToTensor()) 

    encoder = ActionEncoder(
        args.action_dim, args.latent_dim, 256, 3).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder))
    vae = VAE(32, args.device).to(args.device)
    vae.load_state_dict(torch.load(args.vae))

    model = ActionGenerator(11, args.timesteps).to(args.device)
    optimizer = optim.Adam(model.parameters())

    train(model, encoder, vae.decoder, dataset, optimizer, args.timesteps)
