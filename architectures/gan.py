import torch
import torch.nn as nn

from .vae import Upsample


class Generator(nn.Module):
    def __init__(self, action_dim=8):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(action_dim, 1024, 2, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(1024),

            Upsample(2, 1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),

            Upsample(2, 512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            Upsample(2, 256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            Upsample(2, 128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            Upsample(2, 64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x[..., None, None]

        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, action_dim=8):
        super(Discriminator, self).__init__()

        self.action_dim = action_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 1024, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(1024),
        )

        self.action_layers = nn.Sequential(
            nn.Linear(action_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1),
        )

    def forward(self, x, y):
        x = self.conv_layers(x)
        x = x.view(len(x), -1)

        y = self.action_layers(y)

        z = torch.cat([x, y], dim=-1)

        return self.fc(z)


if __name__ == "__main__":
    generator = Generator(8)
    discriminator = Discriminator(8)

    action = torch.rand(13, 8)

    image = generator(action)
    prediction = discriminator(image, action)

    print(image.shape, prediction.shape)
