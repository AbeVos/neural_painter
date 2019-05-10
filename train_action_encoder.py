import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset.dataset import BrushStrokeDataset
from dataset.transform import ToTensor


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim, hidden_layers):
        super(ActionEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU()
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, latent_dim)
        )

    def forwad(self, x):
        return self.layers(x)


def train(dataset, model, optimizer, batch_size=128):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for idx, batch in enumerate(dataloader):
        pass


if __name__ == "__main__":
    dataset = BrushStrokeDataset('labels.csv', 'images/',
                                 transform=ToTensor())

    model = ActionEncoder()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(dataset, model, optimizer)
