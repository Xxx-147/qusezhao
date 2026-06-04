from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "AI model support requires PyTorch. Install it with: python -m pip install -e .[ml]"
        ) from exc
    return torch


def build_model(base_channels: int = 32):
    torch = require_torch()
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(4, channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(4, channels),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return torch.nn.functional.silu(x + self.block(x))

    class FilmMaskNet(nn.Module):
        """Compact residual U-Net for negative-to-positive image translation."""

        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.enc1 = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(c))
            self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
            self.enc2 = nn.Sequential(nn.SiLU(inplace=True), ResidualBlock(c * 2))
            self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
            self.mid = nn.Sequential(nn.SiLU(inplace=True), ResidualBlock(c * 4), ResidualBlock(c * 4))
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
            self.dec2 = nn.Sequential(nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(c * 2))
            self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
            self.dec1 = nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(c))
            self.out = nn.Sequential(nn.Conv2d(c, 3, 3, padding=1), nn.Sigmoid())

        def forward(self, x):  # type: ignore[no-untyped-def]
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            mid = self.mid(self.down2(e2))
            d2 = self.up2(mid)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1)

    return FilmMaskNet()
