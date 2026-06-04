from __future__ import annotations

import argparse
from pathlib import Path

from .data import NegativePairDataset
from .model import build_model, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(prog="film-mask-train", description="Train the AI negative-to-positive model.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    train(args)


def train(args: argparse.Namespace) -> None:
    torch = require_torch()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    dataset = NegativePairDataset(args.manifest, crop_size=args.crop_size, augment=True)
    if len(dataset) == 0:
        raise ValueError("manifest has no training pairs")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_model(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l1 = torch.nn.L1Loss()
    ssim_weight = 0.15

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        iterator = iter(loader)
        for step in range(args.steps_per_epoch):
            try:
                negative, target = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                negative, target = next(iterator)
            negative = negative.to(device)
            target = target.to(device)
            prediction = model(negative)
            loss = l1(prediction, target) + ssim_weight * (1.0 - _simple_ssim(prediction, target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        mean_loss = total_loss / max(args.steps_per_epoch, 1)
        print(f"epoch={epoch} loss={mean_loss:.5f} device={device}")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": mean_loss,
                "base_channels": args.base_channels,
            },
            args.output,
        )


def _simple_ssim(x, y):  # type: ignore[no-untyped-def]
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = x.mean(dim=(-1, -2), keepdim=True)
    mu_y = y.mean(dim=(-1, -2), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-1, -2), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.mean()


if __name__ == "__main__":
    main()
