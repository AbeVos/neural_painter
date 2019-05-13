import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from models import ActionEncoder, VAE
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


def train(dataset, vae, model, optimizer, device='cuda:0', epochs=100,
          batch_size=128):
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        mean_loss = []

        for idx, batch in enumerate(dataloader):
            images, params = batch['image'], batch['parameters']
            images = images.to(device).float()
            params = params.to(device).float()

            code, _ = vae.encoder(images)
            code = code.detach()
            pred = model(params)

            loss = criterion(pred, code)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mean_loss.append(loss.item())

        mean_loss = sum(mean_loss) / len(mean_loss)

        print(f"Epoch {epoch+1}: loss: {mean_loss}")

        recon = torch.sigmoid(vae.decoder(code))
        pred = torch.sigmoid(vae.decoder(pred))

        recon = make_grid(recon[:5], nrow=5)
        pred = make_grid(pred[:5], nrow=5)
        image = make_grid(images[:5], nrow=5)

        plt.figure()
        plt.subplot(311)
        plt.title("Original")
        plt.imshow(image.permute(1, 2, 0).cpu().detach())
        plt.axis('off')
        plt.subplot(312)
        plt.title("Reconstruction")
        plt.imshow(recon.permute(1, 2, 0).cpu().detach())
        plt.axis('off')
        plt.subplot(313)
        plt.title("Neural Painter")
        plt.imshow(pred.permute(1, 2, 0).cpu().detach())
        plt.axis('off')
        plt.savefig(f"samples/recon_{epoch+1:03d}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vae', dest='vae_model', type=str,
        help="Pretrained VAE model.")
    parser.add_argument(
        '--latent_dim', dest='latent_dim', type=int, default=16,
        help="")
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help=""
    )
    parser.add_argument(
        '--device', dest='device', type=str, default='cuda:0',
        help="")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = BrushStrokeDataset('labels.csv', 'images/',
                                 transform=ToTensor())

    vae = VAE(args.latent_dim, device).to(device)
    vae.load_state_dict(torch.load(args.vae_model))
    vae.eval()

    model = ActionEncoder(12, args.latent_dim, 128, 5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(dataset, vae, model, optimizer, device, epochs=args.epochs)

    torch.save(model.state_dict, "models/action_encoder.pth")
