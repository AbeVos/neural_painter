import argparse
import torch
import torch.optim as optim

from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor
from architectures.gan import Generator, Discriminator


def iter_epoch(models, optimizers, dataset, batch_size):
    G, D = models
    optim_G, optim_D = optimizers

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4)

    for batch in dataloader:
        images = batch['image']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    dataset = BrushStrokeDataset('labels.csv', 'images/', transform=ToTensor())

    generator = Generator(args.action_dim, device)
    discriminator = Discriminator(args.action_dim)

    optim_G = optim.Adam(generator.parameters(), lr=3e-4)
    optim_D = optim.Adam(discriminator.parameters(), lr=3e-4)

    train(dataset, model, optimizer, device, args.epochs, args.batch_size,
          args.eval_interval)

    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/gan.pth")
