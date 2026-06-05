from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument("--ssim-weight", type=float, default=0.12)
    parser.add_argument("--chroma-weight", type=float, default=0.32)
    parser.add_argument("--saturation-weight", type=float, default=0.28)
    parser.add_argument("--contrast-weight", type=float, default=0.22)
    parser.add_argument("--gradient-weight", type=float, default=0.18)
    parser.add_argument("--progress-json", type=Path, help="Write epoch metrics to this JSON file.")
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_json or args.output.with_suffix(".progress.json")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {
            "loss": 0.0,
            "pixel_l1": 0.0,
            "ssim": 0.0,
            "chroma": 0.0,
            "saturation": 0.0,
            "contrast": 0.0,
            "gradient": 0.0,
        }
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
            loss, components = _composite_loss(prediction, target, l1, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            for key, value in components.items():
                totals[key] += float(value.detach().cpu())
        mean = {key: value / max(args.steps_per_epoch, 1) for key, value in totals.items()}
        history.append({"epoch": epoch, "device": device, **{key: round(value, 6) for key, value in mean.items()}})
        _write_progress(progress_path, args, history)
        print(
            " ".join(
                [
                    f"epoch={epoch}",
                    f"loss={mean['loss']:.5f}",
                    f"l1={mean['pixel_l1']:.5f}",
                    f"chroma={mean['chroma']:.5f}",
                    f"sat={mean['saturation']:.5f}",
                    f"contrast={mean['contrast']:.5f}",
                    f"grad={mean['gradient']:.5f}",
                    f"device={device}",
                ]
            )
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": mean["loss"],
                "base_channels": args.base_channels,
                "loss_config": _loss_config(args),
            },
            args.output,
        )


def _composite_loss(prediction, target, l1, args):  # type: ignore[no-untyped-def]
    pixel_l1 = l1(prediction, target)
    ssim_loss = 1.0 - _simple_ssim(prediction, target)
    chroma_loss = _opponent_chroma_loss(prediction, target, l1)
    saturation_loss = l1(_saturation(prediction), _saturation(target))
    contrast_loss = l1(_local_contrast(_luminance(prediction)), _local_contrast(_luminance(target)))
    gradient_loss = _gradient_loss(prediction, target, l1)
    loss = (
        pixel_l1
        + args.ssim_weight * ssim_loss
        + args.chroma_weight * chroma_loss
        + args.saturation_weight * saturation_loss
        + args.contrast_weight * contrast_loss
        + args.gradient_weight * gradient_loss
    )
    return loss, {
        "pixel_l1": pixel_l1,
        "ssim": ssim_loss,
        "chroma": chroma_loss,
        "saturation": saturation_loss,
        "contrast": contrast_loss,
        "gradient": gradient_loss,
    }


def _opponent_chroma_loss(x, y, l1):  # type: ignore[no-untyped-def]
    return l1(_opponent_channels(x), _opponent_channels(y))


def _opponent_channels(x):  # type: ignore[no-untyped-def]
    torch = __import__("torch")
    red = x[:, 0:1]
    green = x[:, 1:2]
    blue = x[:, 2:3]
    red_green = red - green
    yellow_blue = (red + green) * 0.5 - blue
    return torch.cat([red_green, yellow_blue], dim=1)


def _saturation(x):  # type: ignore[no-untyped-def]
    max_channel = x.max(dim=1, keepdim=True).values
    min_channel = x.min(dim=1, keepdim=True).values
    return max_channel - min_channel


def _luminance(x):  # type: ignore[no-untyped-def]
    weights = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _local_contrast(luminance):  # type: ignore[no-untyped-def]
    torch = __import__("torch")
    return luminance - torch.nn.functional.avg_pool2d(luminance, kernel_size=7, stride=1, padding=3)


def _gradient_loss(x, y, l1):  # type: ignore[no-untyped-def]
    return l1(_gradient_x(x), _gradient_x(y)) + l1(_gradient_y(x), _gradient_y(y))


def _gradient_x(x):  # type: ignore[no-untyped-def]
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def _gradient_y(x):  # type: ignore[no-untyped-def]
    return x[:, :, 1:, :] - x[:, :, :-1, :]


def _loss_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "ssim_weight": args.ssim_weight,
        "chroma_weight": args.chroma_weight,
        "saturation_weight": args.saturation_weight,
        "contrast_weight": args.contrast_weight,
        "gradient_weight": args.gradient_weight,
    }


def _write_progress(path: Path, args: argparse.Namespace, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(args.manifest),
        "output": str(args.output),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "base_channels": args.base_channels,
        "loss_config": _loss_config(args),
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
