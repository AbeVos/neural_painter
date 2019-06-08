import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim, hidden_layers):
        super(ActionEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(action_dim - 3, hidden_dim),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, latent_dim - 3),
        )

        self.color = nn.Sequential(
            nn.Linear(3, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        color = x[:, 6:9]
        x = torch.cat((x[:, :6], x[:, 9:]), 1)

        x = self.layers(x)
        color = self.color(color)

        return torch.cat((x, color), 1)


class StrokeGenerator(nn.Module):
    def __init__(self, action_dim):
        super(StrokeGenerator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(action_dim, 512, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 512, 5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 5),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 7, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 4, 7, stride=2, padding=2,
                               output_padding=1),
            nn.Tanh(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class StrokeDiscriminator(nn.Module):
    def __init__(self, action_dim):
        super(StrokeDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 7, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.condition = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(inplace=True)
        )

    def forward(self, x, c):
        x = self.conv(x)
        c = self.condition(c)

        x = torch.stack((x.squeeze(), c), -1)

        return self.output(x)


class ActionGenerator(nn.Module):
    def __init__(self, action_dim, timesteps):
        super(ActionGenerator, self).__init__()

        self.timesteps = timesteps

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 7, stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.lstm = nn.LSTM(512, 512, 2, dropout=0.0)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = len(x)

        x = self.conv(x)
        x = x.view(batch_size, -1)

        x = x.unsqueeze(0)

        Y = []
        h = None

        for i in range(self.timesteps):
            y, h = self.lstm(x, h)
            Y.append(y)

        y = torch.stack(Y, dim=0)
        y = y.view(self.timesteps * batch_size, -1)
        y = self.fc(y)

        return y


class PaintingDiscriminator(nn.Module):
    def __init__(self, action_dim):
        super(PaintingDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 7, stride=2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 5),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 1),
        )

    def forward(self, x):
        return self.conv(x).squeeze()
