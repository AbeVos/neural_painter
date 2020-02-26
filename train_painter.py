import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from models.flow import GenerativeLatentFlow
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


def criterion(x, x_recon, y, z, log_det_J, prior):
    log_prob = prior.log_prob(z, y)
    nll = - torch.mean(log_prob + log_det_J)

    recon_error = F.l1_loss(x, x_recon)

    return nll + recon_error


def train(dataset, model, optimizer, device, num_epochs=100,
          batch_size=128, eval_interval=10):
    for epoch in range(num_epochs):
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        )

        last_loss = None

        for it, batch in enumerate(loader):
            model.train()

            x, y = batch.values()
            x = x.float()
            y = y.float()

            x_recon, z, log_det_J = model(x, y)

            loss = criterion(x, x_recon, y, z, log_det_J, model.prior)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if it % eval_interval == 0:
                if last_loss is None:
                    last_loss = loss.item()
                else:
                    last_loss = 0.3 * loss.item() + 0.7 * last_loss

                print(last_loss)


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

    model = GenerativeLatentFlow(100, dim_condition=11)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("trained_models/", exist_ok=True)
    os.makedirs("samples/", exist_ok=True)

    train(dataset, model, optimizer, 'cpu')
