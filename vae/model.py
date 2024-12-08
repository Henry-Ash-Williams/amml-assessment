from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels: int):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=4, stride=2, padding=1
            ),  # (1, 36, 36) => (32, 18, 18)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            ),  # (32, 18, 18) => (64, 9, 9)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # (64, 9, 9) => (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channels: int):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, output_padding=1
            ),  # (128, 4, 4)=> (64, 9, 9)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # (64, 9, 9) => (32, 18, 18)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                32, input_channels, kernel_size=4, stride=2, padding=1
            ),  # (1, 36, 36)
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class VAE(nn.Module):
    """
    A Convolutional Variational Autoencoder (VAE) model.

    Args:
        input_channels (int): Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB).
        latent_dim (int): Size of the latent space.
    """

    def __init__(self, input_channels: int, latent_dim: int):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(input_channels)

        self.feature_dim = 128 * (4 * 4)

        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        self.decoder = Decoder(input_channels)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Samples a latent vector z using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def loss_function(
        reconstruction: Tensor,
        input: Tensor,
        mu: Tensor,
        logvar: Tensor,
        kld_weight=0.001,
    ):
        recons_loss = F.mse_loss(input=reconstruction, target=input)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )
        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "reconstruction_loss": recons_loss.detach(),
            "kld_loss": -kld_loss.detach(),
        }

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encodes the input into latent space parameters (mu and logvar).
        """
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)  # Flatten feature maps
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes a latent vector z into reconstructed input.
        """
        batch_size = z.size(0)
        z = self.fc_decode(z)
        z = z.view(batch_size, 128, 4, 4)
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Runs the VAE forward pass: encodes, reparameterizes, and decodes.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


if __name__ == "__main__":
    test = torch.randn((1, 1, 36, 36))
    net = VAE(1, 100)
    reconstruction, mu, logvar = net(test)
    print(reconstruction.shape, mu.shape, logvar.shape)
