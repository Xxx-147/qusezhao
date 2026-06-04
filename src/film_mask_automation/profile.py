from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ColorProfile:
    matrix: list[list[float]]
    intercept: list[float]
    source: str = "paired-reference"


def fit_color_profile(source_image: Image.Image, reference_image: Image.Image, sample_size: int = 600) -> ColorProfile:
    source = _image_array(source_image, sample_size)
    reference = _image_array(reference_image.resize(source_image.size), sample_size)

    x = source.reshape(-1, 3)
    y = reference.reshape(-1, 3)

    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    matrix = coefficients[:3, :].T
    intercept = coefficients[3, :]

    return ColorProfile(
        matrix=[[round(float(value), 6) for value in row] for row in matrix],
        intercept=[round(float(value), 6) for value in intercept],
    )


def apply_color_profile(image: Image.Image, profile: ColorProfile) -> Image.Image:
    data = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    matrix = np.asarray(profile.matrix, dtype=np.float32)
    intercept = np.asarray(profile.intercept, dtype=np.float32)
    corrected = data @ matrix.T + intercept
    return Image.fromarray(np.clip(corrected * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def save_color_profile(profile: ColorProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2, ensure_ascii=False), encoding="utf-8")


def load_color_profile(path: Path) -> ColorProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ColorProfile(
        matrix=payload["matrix"],
        intercept=payload["intercept"],
        source=payload.get("source", "paired-reference"),
    )


def compare_images(output_path: Path, reference_path: Path) -> dict[str, object]:
    with Image.open(output_path) as output_image, Image.open(reference_path) as reference_image:
        output = np.asarray(output_image.convert("RGB"), dtype=np.float32)
        reference = np.asarray(reference_image.convert("RGB").resize(output_image.size), dtype=np.float32)

    diff = np.abs(output - reference)
    return {
        "mean_absolute_error_rgb": [round(float(value), 2) for value in diff.reshape(-1, 3).mean(axis=0)],
        "mean_absolute_error_total": round(float(diff.mean()), 2),
        "output_size": list(output_image.size),
    }


def _image_array(image: Image.Image, max_size: int) -> np.ndarray:
    rgb = image.convert("RGB")
    rgb.thumbnail((max_size, max_size))
    return np.asarray(rgb, dtype=np.float32) / 255.0
