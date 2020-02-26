import os
import argparse
import torch.optim as optim

from torch.utils.data import DataLoader

from models.flow import GenerativeLatentFlow
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


def train(dataset, model, optimizer, device, num_epochs=100,
          batch_size=128, eval_interval=10):
    for epoch in range(num_epochs):
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        )

        for idx, batch in enumerate(loader):
            model.train()

            images = batch['image']
            images = images.to(device).float()

            print(batch['action'])



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

    model = GenerativeLatentFlow(100, 8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("trained_models/", exist_ok=True)
    os.makedirs("samples/", exist_ok=True)

    train(dataset, model, optimizer, 'cpu')
