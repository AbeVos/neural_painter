import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.flow import GenerativeLatentFlow
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor
from perceptual_loss import _VGGDistance


DIM_ACTION = 8
device = 'cuda:0'


def criterion(x, x_recon, y, z, log_det_J, prior, perceptual_loss):
    log_prob = prior.log_prob(z, y)
    nll = - (log_prob + log_det_J).mean() / z.shape[-1]

    return nll, perceptual_loss(x, x_recon)


def train(dataset, model, perceptual_loss, optimizers, device, num_epochs=100,
          batch_size=512, eval_interval=10):
    fixed_y = torch.rand(32, DIM_ACTION).to(device)

    optim_ae, optim_nf = optimizers

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        )

        last_loss = None

        nll_loss_total = 0
        recon_loss_total = 0

        for it, batch in enumerate(loader, start=1):
            model.train()

            x, y = batch.values()
            x = x[:, -1].unsqueeze(1).float().to(device)
            y = y.float().to(device)
            # y = torch.zeros((len(x), DIM_ACTION)).float().to(device)
            y = y[:, :DIM_ACTION]

            x_recon, z, log_det_J = model(x, y)

            nll_loss, recon_loss = criterion(
                x, x_recon, y, z, log_det_J, model.prior, perceptual_loss
            )
            loss = nll_loss + recon_loss

            loss.backward()
            optim_ae.step()
            optim_nf.step()

            optim_ae.zero_grad()
            optim_nf.zero_grad()
            perceptual_loss.zero_grad()

            nll_loss_total += nll_loss.item()
            recon_loss_total += recon_loss.item()

            if it % eval_interval == 0:
                model.eval()
                images = model.sample(y[:32])
                # images = model.sample(y[:64])
                save_image(
                    torch.cat((images, x_recon[:32], x[:32]), dim=0),
                   "samples.png",
                    pad_value=1
                )
                # save_image(images, "samples.png", pad_value=1)

                nll_loss = nll_loss_total / eval_interval
                recon_loss = recon_loss_total / eval_interval

                nll_loss_total = 0
                recon_loss_total = 0

                print(nll_loss, recon_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', dest='root', default='data'
    )
    args = parser.parse_args()

    dataset = BrushStrokeDataset(
        os.path.join(args.root, "labels.csv"),
        os.path.join(args.root, "images/"),
        transform=ToTensor(),
    )

    model = GenerativeLatentFlow(200, dim_condition=DIM_ACTION, device=device)
    perceptual_loss = _VGGDistance(3).to(device)
    optim_ae = optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters()),
        lr=1e-3
    )
    optim_nf = optim.Adam(
        model.flow.parameters(),
        lr=1e-3,
        weight_decay=2e-5,
    )

    os.makedirs("trained_models/", exist_ok=True)
    os.makedirs("samples/", exist_ok=True)

    train(
        dataset, model, perceptual_loss,
        (optim_ae, optim_nf), device, batch_size=256
    )
