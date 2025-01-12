from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation,
        frame_size,
    ):

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.uniform_(self.W_ci, a=-0.1, b=0.1)
        nn.init.uniform_(self.W_co, a=-0.1, b=0.1)
        nn.init.uniform_(self.W_cf, a=-0.1, b=0.1)

    def forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        output_gate = torch.sigmoid(o_conv + self.W_co * C)
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation,
        image_size,
        device,
    ):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.device = device

        self.convLSTMcell = ConvLSTMCell(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            image_size,
        )

    def forward(self, X):
        # (B, C, L, H, W)
        # B: Batch size
        # C: Channels
        # L: Sequence Length
        # H, W: Width and height

        batch_size, _, seq_len, height, width = X.size()

        output = torch.zeros(
            batch_size, self.out_channels, seq_len, height, width, device=self.device
        )

        H = torch.zeros(
            batch_size, self.out_channels, height, width, device=self.device
        )
        C = torch.zeros(
            batch_size, self.out_channels, height, width, device=self.device
        )

        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)
            output[:, :, time_step] = H

        return output


class ImagePredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_kernels: int,
        num_layers: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        activation: Literal["relu", "tanh"],
        image_size: Tuple[int, int],
        device: Optional[str] = None,
    ):
        super(ImagePredictor, self).__init__()

        if device is None:
            self.device = torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        first_layer = nn.Sequential(
            ConvLSTM(
                in_channels=in_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                image_size=image_size,
                device=self.device,
            ),
            nn.BatchNorm3d(num_features=num_kernels),
        )

        self.layers = nn.Sequential(
            first_layer,
            *[
                nn.Sequential(
                    ConvLSTM(
                        in_channels=num_kernels,
                        out_channels=num_kernels,
                        kernel_size=kernel_size,
                        padding=padding,
                        activation=activation,
                        image_size=image_size,
                        device=self.device,
                    ),
                    nn.BatchNorm3d(num_features=num_kernels),
                )
                for _ in range(2, num_layers + 1)
            ],
        )

        self.conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        output = self.layers(x)
        output = F.sigmoid(self.conv(output[:, :, -1]))

        return output


if __name__ == "__main__":
    device = torch.device("mps")
    model = ImagePredictor(
        1, 64, 5, (3, 3), (1, 1), "relu", (36, 36), device=device
    ).to(device)
    input = torch.randn(64, 1, 2, 36, 36, dtype=torch.float32).to(device)
    output = model(input)
    print(output.min(), output.max())
