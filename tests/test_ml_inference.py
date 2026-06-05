import numpy as np
from PIL import Image

from film_mask_automation.ml.inference import enhance_model_output


def test_enhance_model_output_increases_low_saturation() -> None:
    image = Image.new("RGB", (64, 64), (120, 118, 116))
    image.paste((136, 116, 112), (0, 0, 32, 64))
    image.paste((112, 122, 136), (32, 0, 64, 64))

    enhanced = enhance_model_output(image)

    assert _mean_saturation(enhanced) > _mean_saturation(image)


def _mean_saturation(image: Image.Image) -> float:
    data = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return float((data.max(axis=2) - data.min(axis=2)).mean())
