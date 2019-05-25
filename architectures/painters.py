import torch
import torch.nn as nn

from vae import VAE, Decoder


class VAEPainter(VAE):
    def __init__(self, latent_dim, device, beta=1):
        super(VAEPainter, self).__init__(latent_dim, device, beta)


