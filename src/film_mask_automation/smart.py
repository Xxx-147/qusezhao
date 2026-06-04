from __future__ import annotations

from dataclasses import asdict

import numpy as np
from PIL import Image

from .processor import ConversionParams, ConversionResult, convert_image


SMART_PRESETS: tuple[ConversionParams, ...] = (
    ConversionParams(mask_source="auto", white_percentile=99.0, gamma=1.0, contrast=1.0, white_balance="grayworld"),
    ConversionParams(mask_source="auto", white_percentile=97.0, gamma=1.25, contrast=0.85, white_balance="none"),
    ConversionParams(mask_source="auto", white_percentile=95.0, gamma=1.55, contrast=0.72, white_balance="none"),
    ConversionParams(mask_source="border", white_percentile=97.0, gamma=1.3, contrast=0.82, white_balance="none"),
    ConversionParams(mask_source="percentile", white_percentile=99.0, gamma=1.05, contrast=0.95, white_balance="grayworld"),
    ConversionParams(mask_source="percentile", white_percentile=95.0, gamma=1.7, contrast=0.65, white_balance="none"),
)


def convert_image_smart(image: Image.Image) -> ConversionResult:
    preview_source = image.convert("RGB")
    preview_source.thumbnail((420, 420))

    scored: list[tuple[float, ConversionParams, ConversionResult]] = []
    for params in SMART_PRESETS:
        result = convert_image(preview_source, params)
        score = _score_candidate(result.image)
        scored.append((score, params, result))

    best_score, best_params, _ = min(scored, key=lambda item: item[0])
    full_result = convert_image(image, best_params)
    diagnostics = dict(full_result.diagnostics)
    diagnostics["smart_auto"] = {
        "score": round(float(best_score), 4),
        "selected_params": asdict(best_params),
        "candidate_count": len(SMART_PRESETS),
    }
    return ConversionResult(image=full_result.image, diagnostics=diagnostics)


def _score_candidate(image: Image.Image) -> float:
    data = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pixels = data.reshape(-1, 3)
    luminance = pixels @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

    mean_luma = float(np.mean(luminance))
    std_luma = float(np.std(luminance))
    low_clip = float(np.mean(luminance < 0.025))
    high_clip = float(np.mean(luminance > 0.975))

    means = np.mean(pixels, axis=0)
    blue_cyan_cast = max(0.0, float(means[2] - means[0])) + max(0.0, float(means[1] - means[0]) * 0.35)
    red_orange_cast = max(0.0, float(means[0] - means[2]) - 0.18)
    saturation = np.mean(np.max(pixels, axis=1) - np.min(pixels, axis=1))

    score = 0.0
    score += abs(mean_luma - 0.52) * 1.8
    score += abs(std_luma - 0.22) * 1.2
    score += low_clip * 4.0
    score += high_clip * 3.0
    score += blue_cyan_cast * 1.4
    score += red_orange_cast * 0.8
    score += max(0.0, 0.08 - float(saturation)) * 0.8
    score += max(0.0, float(saturation) - 0.42) * 0.5
    return score
