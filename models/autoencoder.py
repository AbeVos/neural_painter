import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dim_z):
        super(Encoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, 5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, dim_z, 5, stride=2),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        return x.view(len(x), -1)


class Decoder(nn.Module):
    def __init__(self, dim_z):
        super(Decoder, self).__init__()

        self.tconv_layers = nn.Sequential(
            nn.ConvTranspose2d(dim_z, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 1, 5, stride=2, output_padding=1),
            nn.Sigmoid(),
        )

        self.dim_z = dim_z

    def forward(self, x):
        x = x.view(len(x), self.dim_z, 1, 1)
        return self.tconv_layers(x)
