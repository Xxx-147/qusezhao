from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from .model import build_model, require_torch


def convert_with_model(
    image: Image.Image,
    checkpoint_path: Path,
    device: str = "cpu",
    enhance: bool = True,
    hybrid_anchor: bool = True,
) -> Image.Image:
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
    result = result.resize((width, height))
    if not enhance:
        return result
    enhanced = enhance_model_output(result)
    if hybrid_anchor:
        enhanced = _blend_with_smart_anchor(source, enhanced)
    return enhanced


def enhance_model_output(image: Image.Image, target_saturation: float = 0.16, target_luma_std: float = 0.18) -> Image.Image:
    """Conservative color restoration for small models that regress to gray outputs."""
    rgb = _neutralize_channel_cast(image.convert("RGB"))
    data = np.asarray(rgb, dtype=np.float32) / 255.0
    saturation = float((data.max(axis=2) - data.min(axis=2)).mean())
    luminance = data @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_std = float(luminance.std())

    color_factor = _bounded_factor(target_saturation, saturation, minimum=1.0, maximum=2.35)
    contrast_factor = _bounded_factor(target_luma_std, luma_std, minimum=1.0, maximum=1.28)
    enhanced = ImageEnhance.Color(rgb).enhance(color_factor)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)
    return enhanced


def _neutralize_channel_cast(image: Image.Image, strength: float = 0.58) -> Image.Image:
    data = np.asarray(image, dtype=np.float32)
    luminance = data @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mask = (luminance > np.percentile(luminance, 15)) & (luminance < np.percentile(luminance, 92))
    if not np.any(mask):
        mask = np.ones(luminance.shape, dtype=bool)
    means = data[mask].reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1.0), 0.72, 1.34)
    gains = 1.0 + (gains - 1.0) * strength
    corrected = np.clip(data * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB")


def _bounded_factor(target: float, current: float, minimum: float, maximum: float) -> float:
    if current <= 1e-6:
        return maximum
    return max(minimum, min(maximum, target / current))


def _blend_with_smart_anchor(source: Image.Image, model_output: Image.Image, model_weight: float = 0.55) -> Image.Image:
    from film_mask_automation.smart import convert_image_smart

    smart = convert_image_smart(source).image.resize(model_output.size)
    blended = Image.blend(smart.convert("RGB"), model_output.convert("RGB"), model_weight)
    blended = ImageEnhance.Color(blended).enhance(1.35)
    return ImageEnhance.Contrast(blended).enhance(1.12)


def _pad_to_multiple(image: Image.Image, multiple: int) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    padded_width = ((width + multiple - 1) // multiple) * multiple
    padded_height = ((height + multiple - 1) // multiple) * multiple
    if padded_width == width and padded_height == height:
        return image, image.size
    canvas = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas, image.size
