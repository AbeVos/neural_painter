import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from models import VAE, ActionEncoder, ActionGenerator, PaintingDiscriminator


def blend(src, dst):
    alpha = src[:, -1, ...].unsqueeze(1)
    src = src[:, :-1, ...]

    return alpha * src + (1 - alpha) * dst


def train(agent, generator, discriminator, optimizer_G, optimizer_D,
          dataset, timesteps=20):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16,
                            num_workers=4)

    loss_g_plot = []
    loss_d_plot = []

    criterion = nn.BCELoss()

    for idx, (images, _) in enumerate(dataloader):
        images = images.to(args.device)

        ones = torch.ones(len(images)).to(args.device)
        zeros = torch.zeros(len(images)).to(args.device)

        # Train Agent.
        canvas = torch.ones_like(images).to(args.device)[:, :3]

        code = agent(images)
        strokes = generator(code).view(timesteps, len(images), 4, 64, 64)

        # Blend the strokes into a single painting.
        for step, stroke in enumerate(strokes):
            canvas = blend(stroke, canvas)

        pred_painting = discriminator(canvas).squeeze()
        loss_g = criterion(pred_painting, ones)

        loss_g.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()

        loss_g_plot.append(loss_g.item())

        if idx % 50 == 0:
            results = torch.cat((canvas[:5], images[:5]), 0)

            print(f"Epoch {idx} | Agent Loss: {loss_g.item()}")
            save_image(results, f"samples/painting_{idx:05d}.png", nrow=5)

            plt.figure()
            plt.plot(loss_g_plot, label="Action Generator Loss")
            plt.plot(loss_d_plot, label="Discriminator Loss")
            plt.savefig("action_generator_loss.png")
            plt.close()


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

    dataset = ImageFolder('train_64x64/', transform=transforms.ToTensor()) 

    encoder = ActionEncoder(
        args.action_dim, args.latent_dim, 256, 5).to(args.device)
    encoder.load_state_dict(torch.load(args.encoder))
    vae = VAE(args.latent_dim, args.device).to(args.device)
    vae.load_state_dict(torch.load(args.vae))

    agent = ActionGenerator(args.action_dim, args.timesteps).to(args.device)
    optimizer_G = optim.Adam(agent.parameters())

    generator = nn.Sequential(encoder, vae.decoder)

    discriminator = PaintingDiscriminator(args.action_dim).to(args.device)
    optimizer_D = optim.Adam(discriminator.parameters())

    train(agent, generator, discriminator, optimizer_G, optimizer_D,
          dataset, args.timesteps)
