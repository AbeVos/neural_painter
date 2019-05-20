import argparse
import torch

from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor
from models import StrokeGenerator, StrokeDiscriminator


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

    generator = StrokeGenerator(args.action_dim, device)
    discriminator = StrokeDiscriminator(args.action_dim)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train(dataset, model, optimizer, device, args.epochs, args.batch_size,
          args.eval_interval)

    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/gan.pth")
