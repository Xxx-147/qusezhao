from __future__ import annotations

from pathlib import Path

from PIL import Image

from film_mask_automation.profile import apply_color_profile, compare_images, fit_color_profile, save_color_profile
from film_mask_automation.processor import ConversionParams, convert_file


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    negative_path = project_root / "samples" / "input" / "metro_negative.jpg"
    reference_path = project_root / "samples" / "reference" / "metro_reference.jpg"
    output_path = project_root / "samples" / "output" / "metro_positive.jpg"
    profiled_output_path = project_root / "samples" / "output" / "metro_positive_profiled.jpg"
    profile_path = project_root / "profiles" / "metro_profile.json"

    diagnostics = convert_file(
        negative_path,
        output_path,
        ConversionParams(mask_source="percentile", contrast=1.12, gamma=1.04, white_balance="grayworld"),
    )
    metrics = compare_images(output_path, reference_path)
    with Image.open(output_path) as output_image, Image.open(reference_path) as reference_image:
        profile = fit_color_profile(output_image, reference_image)
        save_color_profile(profile, profile_path)
        apply_color_profile(output_image, profile).save(profiled_output_path)

    profiled_metrics = compare_images(profiled_output_path, reference_path)
    print(
        {
            "output": str(output_path),
            "profiled_output": str(profiled_output_path),
            "profile": str(profile_path),
            "diagnostics": diagnostics,
            "metrics": metrics,
            "profiled_metrics": profiled_metrics,
        }
    )


if __name__ == "__main__":
    main()
