import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from models import VAE, ActionEncoder, ActionGenerator, PaintingDiscriminator


def blend(src, dst):
    alpha = src[:, -1, ...].unsqueeze(1)
    src = src[:, :-1, ...]

    return alpha * src + (1 - alpha) * dst


def paint(images, agent, generator, timesteps):
    canvas = torch.ones_like(images).to(args.device)[:, :3]

    codes = agent(images)
    strokes = generator(codes).view(timesteps, len(images), 4, 64, 64)

    for step, stroke in enumerate(strokes):
        canvas = blend(stroke, canvas)

    return canvas


def enable_gradients(net):
    for p in net.parameters():
        p.requires_grad = True


def disable_gradients(net):
    for p in net.parameters():
        p.requires_grad = False


def train(agent, generator, discriminator, optimizer_G, optimizer_D,
          dataset, timesteps=20):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16,
                            num_workers=4)

    loss_g_plot = []
    loss_d_plot = []

    criterion = nn.BCELoss()

    one = torch.FloatTensor([1]).to(args.device)
    mone = -1 * one

    for idx, (images, _) in enumerate(dataloader):
        enable_gradients(discriminator)
        disable_gradients(agent)

        images = images.to(args.device)

        ones = torch.ones(len(images)).to(args.device)
        zeros = torch.zeros(len(images)).to(args.device)

        ########################
        # Train Discriminator. #
        ########################

        painting = paint(images, agent, generator, timesteps)

        pred_real = discriminator(images)
        pred_fake = discriminator(painting)

        # loss_d = criterion(pred_images, ones) \
        #     + criterion(pred_painting, zeros)

        loss_d = pred_fake.mean() - pred_real.mean()

        alpha = torch.rand(images.shape).to(args.device)

        interpolation = alpha * images + (1 - alpha) * painting
        pred_interp = discriminator(interpolation)

        gradients = grad(outputs=pred_interp.sum(), inputs=interpolation,
                         create_graph=True)[0]
        penalty = ((
            gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1
        ) ** 2).mean()

        loss_d += 10 * penalty

        loss_d.backward()
        optimizer_D.step()
        optimizer_D.zero_grad()

        loss_d_plot.append(loss_d.item())

        ################
        # Train Agent. #
        ################

        if idx % 5 == 0:
            enable_gradients(agent)
            disable_gradients(discriminator)

            painting = paint(images, agent, generator, timesteps)

            pred_fake = discriminator(painting)
            # loss_g = criterion(pred_painting, ones)
            loss_g = - pred_fake.mean()

            loss_g.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

        loss_g_plot.append(loss_g.item())

        if idx % 50 == 0:
            results = torch.cat((painting[:5], images[:5]), 0)

            print(f"Epoch {idx} | Agent loss: {loss_g.item()}, "
                  f"Critic loss: {loss_d.item()}")
            save_image(results, f"samples/painting_{idx:05d}.png", nrow=5)

            plt.figure()
            plt.plot(loss_g_plot, label="Action Generator Loss")
            plt.plot(loss_d_plot, label="Discriminator Loss")
            plt.legend()
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
        '--latent_dim', dest='latent_dim', type=int, default=20,
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

    generator = nn.Sequential(encoder, vae.decoder)

    discriminator = PaintingDiscriminator(args.action_dim).to(args.device)

    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=5e-5)
    optimizer_G = optim.RMSprop(agent.parameters(), lr=5e-5)

    train(agent, generator, discriminator, optimizer_G, optimizer_D,
          dataset, args.timesteps)
