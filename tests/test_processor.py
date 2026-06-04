from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from film_mask_automation.processor import ConversionParams, convert_file, convert_image


def test_convert_image_estimates_border_mask() -> None:
    image = make_synthetic_negative()
    result = convert_image(image, ConversionParams(mask_source="border", contrast=1.0, white_balance="none"))

    assert result.image.mode == "RGB"
    assert result.diagnostics["mask_rgb"] == [232, 152, 82]


def test_convert_file_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "negative.jpg"
    output_path = tmp_path / "positive.jpg"
    make_synthetic_negative().save(input_path)

    diagnostics = convert_file(input_path, output_path, ConversionParams(mask_source="border"))

    assert output_path.exists()
    assert diagnostics["mask_rgb"][0] >= 220


def make_synthetic_negative() -> Image.Image:
    width, height = 480, 320
    positive = Image.new("RGB", (width, height), (180, 190, 180))
    draw = ImageDraw.Draw(positive)
    draw.rectangle((70, 60, 220, 260), fill=(205, 85, 65))
    draw.rectangle((250, 50, 420, 150), fill=(70, 145, 210))
    draw.ellipse((260, 170, 410, 285), fill=(235, 220, 170))

    pos = np.asarray(positive, dtype=np.float32) / 255.0
    mask = np.asarray([232, 152, 82], dtype=np.float32) / 255.0
    black = np.asarray([22, 18, 16], dtype=np.float32) / 255.0
    negative = black + (1.0 - pos) * (mask - black)
    negative_image = Image.fromarray(np.clip(negative * 255, 0, 255).astype(np.uint8), "RGB")

    canvas = Image.new("RGB", (width + 60, height + 60), tuple((mask * 255).astype(int)))
    canvas.paste(negative_image, (30, 30))
    return canvas
