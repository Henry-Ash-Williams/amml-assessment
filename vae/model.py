import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, z):
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        return z


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dims: list[int] = None):
        super(VAE, self).__init__()

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.latent_dim = latent_dim

        modules = []
        in_channels = 1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 5 * 5, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 5 * 5, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 5 * 5)

        self.decoder = Decoder()
        self.final_layer = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, 128, 5, 5)  # Reshape for deconvolution
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    @staticmethod
    def loss_fn(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kld_weight,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + (kld_weight * kl_div)


# Example usage
if __name__ == "__main__":
    vae = VAE(latent_dim=16)
    sample_input = torch.randn(8, 1, 36, 36)  # Batch of 8 grayscale images
    recon_output, mu, log_var = vae(sample_input)
    print(f"Reconstruction Shape: {recon_output.shape}")
    loss = vae.loss_fn(recon_output, sample_input, mu, log_var)
    print(f"Reconstruction Loss: {loss.item()}")
