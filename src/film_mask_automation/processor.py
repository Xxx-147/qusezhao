from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageOps

MaskSource = Literal["auto", "border", "percentile", "manual"]
WhiteBalance = Literal["grayworld", "none"]


@dataclass(frozen=True)
class ConversionParams:
    mask_source: MaskSource = "auto"
    manual_mask_rgb: tuple[float, float, float] | None = None
    border_fraction: float = 0.06
    black_percentile: float = 0.5
    white_percentile: float = 99.5
    reference_exponent: float = 1.0
    red_ratio: float = 1.0
    blue_ratio: float = 1.0
    exposure: float = 0.0
    brightness: float = 0.0
    gamma: float = 1.0
    contrast: float = 1.08
    saturation: float = 1.0
    temperature: float = 0.0
    tint: float = 0.0
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    white_balance: WhiteBalance = "grayworld"
    sharpen: float = 0.0


@dataclass(frozen=True)
class ConversionResult:
    image: Image.Image
    diagnostics: dict[str, object]


def convert_image(image: Image.Image, params: ConversionParams | None = None) -> ConversionResult:
    params = params or ConversionParams()
    _validate_params(params)

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    data = np.asarray(rgb, dtype=np.float32) / 255.0

    mask = _estimate_mask(data, params)
    black = _estimate_black(data, mask, params.black_percentile)

    positive = _remove_mask_and_invert(data, black, mask, params)
    positive = _stretch_levels(positive, low_pct=1.0, high_pct=params.white_percentile)
    positive = _apply_exposure_and_brightness(positive, params.exposure, params.brightness)
    positive = _apply_gamma(positive, params.gamma)
    positive = _apply_contrast(positive, params.contrast)
    positive = _apply_saturation(positive, params.saturation)
    positive = _apply_color_temperature_tint(positive, params.temperature, params.tint)
    positive = _apply_rgb_gains(positive, (params.red_gain, params.green_gain, params.blue_gain))

    wb_gains = np.ones(3, dtype=np.float32)
    if params.white_balance == "grayworld":
        positive, wb_gains = _gray_world_balance(positive)

    output = Image.fromarray(np.clip(positive * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    if params.sharpen > 0:
        output = _unsharp(output, params.sharpen)

    return ConversionResult(
        image=output,
        diagnostics={
            "params": asdict(params),
            "mask_rgb": _as_rgb255(mask),
            "black_rgb": _as_rgb255(black),
            "white_balance_gains": [round(float(value), 4) for value in wb_gains],
        },
    )


def convert_file(input_path: Path, output_path: Path, params: ConversionParams | None = None) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        result = convert_image(image, params)
        result.image.save(output_path)
        return result.diagnostics


def _validate_params(params: ConversionParams) -> None:
    if params.mask_source == "manual" and params.manual_mask_rgb is None:
        raise ValueError("manual mask mode requires manual_mask_rgb")
    if not 0.01 <= params.border_fraction <= 0.25:
        raise ValueError("border_fraction must be between 0.01 and 0.25")
    if params.black_percentile < 0 or params.black_percentile >= 25:
        raise ValueError("black_percentile must be between 0 and 25")
    if params.white_percentile <= 50 or params.white_percentile > 100:
        raise ValueError("white_percentile must be between 50 and 100")
    if params.reference_exponent <= 0:
        raise ValueError("reference_exponent must be greater than 0")
    if params.red_ratio <= 0 or params.blue_ratio <= 0:
        raise ValueError("red_ratio and blue_ratio must be greater than 0")
    if params.gamma <= 0:
        raise ValueError("gamma must be greater than 0")
    if params.contrast <= 0:
        raise ValueError("contrast must be greater than 0")
    if params.saturation < 0:
        raise ValueError("saturation must be greater than or equal to 0")
    if params.red_gain <= 0 or params.green_gain <= 0 or params.blue_gain <= 0:
        raise ValueError("RGB gains must be greater than 0")
    if params.sharpen < 0:
        raise ValueError("sharpen must be greater than or equal to 0")


def _estimate_mask(data: np.ndarray, params: ConversionParams) -> np.ndarray:
    if params.mask_source == "manual":
        mask = np.asarray(params.manual_mask_rgb, dtype=np.float32) / 255.0
        return np.clip(mask, 0.01, 1.0)

    if params.mask_source in ("auto", "border"):
        border = _border_pixels(data, params.border_fraction)
        border_mask = _robust_high_median(border)
        if params.mask_source == "border":
            return border_mask

        image_mask = _robust_high_median(data.reshape(-1, 3))
        image_mid = np.percentile(data.reshape(-1, 3), 50, axis=0)
        border_confidence = float(np.mean(border_mask) - np.mean(image_mid))
        return border_mask if border_confidence > 0.08 else image_mask

    return _robust_high_median(data.reshape(-1, 3))


def _estimate_black(data: np.ndarray, mask: np.ndarray, percentile: float) -> np.ndarray:
    pixels = data.reshape(-1, 3)
    low = np.percentile(pixels, percentile, axis=0).astype(np.float32)
    return np.minimum(low, mask - 0.02)


def _remove_mask_and_invert(data: np.ndarray, black: np.ndarray, mask: np.ndarray, params: ConversionParams) -> np.ndarray:
    denominator = np.maximum(mask - black, 0.03)
    normalized_negative = (data - black) / denominator
    exponents = np.asarray(
        [
            params.reference_exponent * params.red_ratio,
            params.reference_exponent,
            params.reference_exponent * params.blue_ratio,
        ],
        dtype=np.float32,
    )
    density = np.power(np.clip(normalized_negative, 0.0, 1.0), exponents)
    return 1.0 - np.clip(density, 0.0, 1.0)


def _stretch_levels(data: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    pixels = data.reshape(-1, 3)
    low = np.percentile(pixels, low_pct, axis=0).astype(np.float32)
    high = np.percentile(pixels, high_pct, axis=0).astype(np.float32)
    denominator = np.maximum(high - low, 0.02)
    return np.clip((data - low) / denominator, 0.0, 1.0)


def _apply_gamma(data: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(data, 0.0, 1.0), 1.0 / gamma)


def _apply_contrast(data: np.ndarray, contrast: float) -> np.ndarray:
    return np.clip((data - 0.5) * contrast + 0.5, 0.0, 1.0)


def _apply_exposure_and_brightness(data: np.ndarray, exposure: float, brightness: float) -> np.ndarray:
    adjusted = data * float(2.0**exposure)
    adjusted = adjusted + brightness
    return np.clip(adjusted, 0.0, 1.0)


def _apply_saturation(data: np.ndarray, saturation: float) -> np.ndarray:
    luminance = np.sum(data * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2, keepdims=True)
    return np.clip(luminance + (data - luminance) * saturation, 0.0, 1.0)


def _apply_color_temperature_tint(data: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    gains = np.asarray(
        [
            1.0 + temperature * 0.18 - tint * 0.03,
            1.0 - abs(temperature) * 0.04 + tint * 0.12,
            1.0 - temperature * 0.18 - tint * 0.03,
        ],
        dtype=np.float32,
    )
    return np.clip(data * np.clip(gains, 0.5, 1.5), 0.0, 1.0)


def _apply_rgb_gains(data: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(data * np.asarray(gains, dtype=np.float32), 0.0, 1.0)


def _gray_world_balance(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = data.reshape(-1, 3)
    valid = pixels[(pixels.max(axis=1) < 0.98) & (pixels.min(axis=1) > 0.02)]
    sample = valid if len(valid) > 100 else pixels
    means = np.maximum(np.mean(sample, axis=0), 0.001)
    target = float(np.mean(means))
    gains = np.clip(target / means, 0.5, 2.0).astype(np.float32)
    return np.clip(data * gains, 0.0, 1.0), gains


def _border_pixels(data: np.ndarray, fraction: float) -> np.ndarray:
    height, width, _ = data.shape
    edge = max(1, int(min(height, width) * fraction))
    strips = [
        data[:edge, :, :],
        data[-edge:, :, :],
        data[:, :edge, :],
        data[:, -edge:, :],
    ]
    return np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)


def _robust_high_median(pixels: np.ndarray) -> np.ndarray:
    orange_base = _orange_base_candidates(pixels)
    if len(orange_base) >= 64:
        return np.clip(np.median(orange_base, axis=0).astype(np.float32), 0.05, 1.0)

    brightness = pixels.mean(axis=1)
    not_paper_white = pixels.max(axis=1) < 0.97
    usable = pixels[not_paper_white]
    if len(usable) < 64:
        usable = pixels
        brightness = usable.mean(axis=1)
    else:
        brightness = usable.mean(axis=1)
    threshold = np.percentile(brightness, 85)
    bright_pixels = usable[brightness >= threshold]
    if len(bright_pixels) < 16:
        bright_pixels = usable
    return np.clip(np.median(bright_pixels, axis=0).astype(np.float32), 0.05, 1.0)


def _orange_base_candidates(pixels: np.ndarray) -> np.ndarray:
    red = pixels[:, 0]
    green = pixels[:, 1]
    blue = pixels[:, 2]
    brightness = pixels.mean(axis=1)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    orange_like = (
        (red > green * 1.04)
        & (green > blue * 1.02)
        & (brightness > 0.20)
        & (brightness < 0.92)
        & (chroma > 0.06)
        & (pixels.max(axis=1) < 0.98)
    )
    candidates = pixels[orange_like]
    if len(candidates) < 64:
        return candidates
    candidate_brightness = candidates.mean(axis=1)
    high = np.percentile(candidate_brightness, 65)
    return candidates[candidate_brightness >= high]


def _unsharp(image: Image.Image, amount: float) -> Image.Image:
    percent = int(np.clip(amount, 0.0, 2.0) * 150)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=percent, threshold=3))


def _as_rgb255(value: np.ndarray) -> list[int]:
    return [int(round(float(channel) * 255)) for channel in value]
