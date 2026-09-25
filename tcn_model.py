"""
Small dilated causal CNN (TCN-style) forecaster -- designed from scratch to
comfortably fit STM32F446RE (512 KB flash, 128 KB RAM).

This is the "from-scratch baseline" for the thesis: same forecasting task as
TTM, but sized specifically for genuinely constrained hardware instead of
being a compressed foundation model.

Usage:
    from tcn_model import SmallTCN
    model = SmallTCN(context_length=128, forecast_length=96)          # 1 channel (original)
    model = SmallTCN(context_length=36, forecast_length=18, in_channels=3)  # solar w/ time features
    out = model(x)  # x: (batch, context_length, in_channels)
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding, so the model never
    sees future timesteps -- required for valid forecasting."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.pad, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        # remove the extra padding added on the right (causal = past-only)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    """One dilated causal conv layer + activation + (optional) norm."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class SmallTCN(nn.Module):
    """
    Small dilated causal CNN forecaster.

    Architecture (default config, ~42K params @ in_channels=1):
        Conv1D k=5 d=1   in_channels -> 32 channels
        Conv1D k=5 d=2  32 -> 32 channels
        Conv1D k=5 d=4  32 -> 64 channels
        Conv1D k=5 d=8  64 -> 64 channels
        Global pooling over time -> Dense head -> forecast_length outputs

    Sized to comfortably fit STM32F446RE (512 KB flash, 128 KB RAM) even
    before quantization, and with wide margin after int8 quantization.

    in_channels: number of features per timestep. Default 1 (raw value
    only, matches the original REFIT/load-data setup). The solar variant
    uses in_channels=3 ([generation, sin(hour), cos(hour)]) -- note this
    changes parameter count slightly (only the first conv layer's input
    dim grows), which should be reported explicitly in hardware benchmarks
    since it's a deliberate deviation from the original REFIT baseline.
    """

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 96,
        channels=(32, 32, 64, 64),
        kernel_size: int = 5,
        in_channels: int = 1,
    ):
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.in_channels = in_channels

        dilations = [2 ** i for i in range(len(channels))]
        in_ch = in_channels
        blocks = []
        for out_ch, dilation in zip(channels, dilations):
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Head: use the last timestep's features (causal -> represents
        # everything seen so far) to predict the full forecast horizon.
        self.head = nn.Linear(in_ch, forecast_length)

    def forward(self, x):
        # x: (batch, context_length, in_channels) -> conv wants (batch, channels, time)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        # take the last timestep (causal representation of full context)
        last = x[:, :, -1]
        out = self.head(last)
        return out  # (batch, forecast_length)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for in_ch in (1, 3):
        model = SmallTCN(context_length=128, forecast_length=96, in_channels=in_ch)
        n_params = count_params(model)
        print(f"SmallTCN (in_channels={in_ch}) parameters: {n_params:,}")
        print(f"  Estimated size @ fp32: {n_params * 4 / 1024:.1f} KB")
        print(f"  Estimated size @ int8: {n_params * 1 / 1024:.1f} KB")

        dummy = torch.randn(1, 128, in_ch)
        out = model(dummy)
        print(f"  Input shape: {dummy.shape} -> Output shape: {out.shape}")
        print()
