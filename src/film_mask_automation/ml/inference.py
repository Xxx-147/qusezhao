from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .model import build_model, require_torch


def convert_with_model(image: Image.Image, checkpoint_path: Path, device: str = "cpu") -> Image.Image:
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    base_channels = checkpoint.get("base_channels", 32) if isinstance(checkpoint, dict) else 32
    model = build_model(base_channels=base_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    source = image.convert("RGB")
    width, height = source.size
    padded, original_size = _pad_to_multiple(source, multiple=4)
    array = np.asarray(padded, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
    result = Image.fromarray((output * 255.0).astype(np.uint8), mode="RGB")
    result = result.crop((0, 0, original_size[0], original_size[1]))
    return result.resize((width, height))


def _pad_to_multiple(image: Image.Image, multiple: int) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    padded_width = ((width + multiple - 1) // multiple) * multiple
    padded_height = ((height + multiple - 1) // multiple) * multiple
    if padded_width == width and padded_height == height:
        return image, image.size
    canvas = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas, image.size
