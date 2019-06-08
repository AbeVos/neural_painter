import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from architectures.painters import ActionEncoder, VAEPainter
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


def create_images(alpha, color, device='cpu'):
    channels = torch.ones((len(color), 3, 64, 64)).to(device)
    channels *= color[..., None, None]

    return torch.cat((channels, alpha), dim=1)


def blend(src, dst):
    dst = dst[:, :3]
    alpha = src[:, -1].unsqueeze(1)
    src = src[:, :3]

    return alpha * src + (1 - alpha) * dst


def save_recon_plot(original, reconstructed, painted, filename):
    background = torch.ones((len(original), 3, 64, 64))
    original = blend(original, background)
    reconstructed = blend(reconstructed, background)
    painted = blend(painted, background)

    original = make_grid(original, nrow=8)
    reconstructed = make_grid(reconstructed, nrow=8)
    painted = make_grid(painted, nrow=8)

    plt.figure()
    plt.subplot(311)
    plt.title("Original")
    plt.imshow(original.permute(1, 2, 0).cpu().detach())
    plt.axis('off')

    plt.subplot(312)
    plt.title("Reconstruction")
    plt.imshow(reconstructed.permute(1, 2, 0).cpu().detach())
    plt.axis('off')

    plt.subplot(313)
    plt.title("Neural Painter")
    plt.imshow(painted.permute(1, 2, 0).cpu().detach())
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


def train(dataset, vae, model, optimizer, device='cuda:0', epochs=100,
          batch_size=64):
    criterion = nn.MSELoss()

    mean_loss = []

    for epoch in range(epochs):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        for idx, batch in enumerate(dataloader):
            images, params = batch['image'], batch['parameters']
            images = images.to(device).float()
            params = params.to(device).float()
            color = params[:, 6:9]
            params = torch.cat((params[:, :6], params[:, 9:]), dim=1)

            code, _ = vae.encoder(images[:, -1].unsqueeze(1))
            code = code.detach()
            pred = model(params)

            loss = criterion(pred, code)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mean_loss.append(loss.item())

            if idx % args.eval_interval == 0:
                mean_loss = sum(mean_loss) / len(mean_loss)

                print(f"Epoch {epoch+1}: loss: {mean_loss}")

                images = images[:8].detach().cpu()

                color = color.detach().cpu()[:8]
                recon = create_images(
                    torch.sigmoid(vae.decoder(code[:8])).detach().cpu(), color)
                pred = create_images(
                    torch.sigmoid(vae.decoder(pred[:8])).detach().cpu(), color)

                save_recon_plot(
                    images, recon, pred, f"samples/recontructions.png")

                mean_loss = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vae', dest='vae_model', type=str,
        help="Pretrained VAE model.")
    parser.add_argument(
        '--action_dim', dest='action_dim', type=int, default=8,
        help="")
    parser.add_argument(
        '--latent_dim', dest='latent_dim', type=int, default=8,
        help="")
    parser.add_argument(
        '--epochs', dest='epochs', type=int, default=100,
        help=""
    )
    parser.add_argument(
        '--eval_interval', type=int, default=100,
        help="")
    parser.add_argument(
        '--data_root', dest='root', default='images_100k',
        help="")
    parser.add_argument(
        '--device', dest='device', type=str, default='cuda:0',
        help="")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = BrushStrokeDataset(os.path.join(args.root, 'labels.csv'),
                                 os.path.join(args.root, 'images/'),
                                              transform=ToTensor())

    vae = VAEPainter(args.latent_dim, device).to(device)
    vae.load_state_dict(torch.load(args.vae_model))
    vae.eval()

    model = ActionEncoder(args.action_dim, args.latent_dim, 256, 5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(dataset, vae, model, optimizer, device, epochs=args.epochs)

    torch.save(model.state_dict(), "models/action_encoder.pth")
