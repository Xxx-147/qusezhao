from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from film_mask_automation.ml.inference import convert_with_model
from film_mask_automation.processor import ConversionParams, convert_image
from film_mask_automation.profile import compare_images
from film_mask_automation.smart import convert_image_smart


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual package for datasets and model outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("review_packages/2026-06-04_model_review"))
    parser.add_argument("--model", type=Path, default=Path("models/film_mask_tiny_blueneg.pt"))
    parser.add_argument("--local-model", type=Path, default=Path("models/film_mask_tiny_local.pt"))
    args = parser.parse_args()

    root = Path.cwd()
    package = args.output_dir
    if not package.is_absolute():
        package = root / package
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)

    train_manifest = root / "datasets" / "blueneg_subset" / "blueneg_train_manifest.csv"
    test_manifest = root / "datasets" / "blueneg_subset" / "blueneg_test_manifest.csv"
    train_records = _read_manifest(train_manifest)
    test_records = _read_manifest(test_manifest)

    shutil.copy2(train_manifest, package / "blueneg_train_manifest.csv")
    shutil.copy2(test_manifest, package / "blueneg_test_manifest.csv")

    _make_pair_sheet(
        train_records[:8],
        train_manifest.parent,
        package / "01_training_set_sheet.jpg",
        title="BlueNeg training set examples",
    )
    _make_pair_sheet(
        test_records,
        test_manifest.parent,
        package / "02_test_set_sheet.jpg",
        title="BlueNeg test set examples",
    )

    metrics: dict[str, object] = {}
    metrics["blueneg_model"] = _make_model_output_sheet(
        test_records,
        test_manifest.parent,
        args.model,
        package / "03_blueneg_model_outputs_sheet.jpg",
        package / "model_outputs_blueneg",
    )
    metrics["local_negative_outputs"] = _make_local_output_sheet(
        root,
        args.local_model,
        package / "04_local_negative_outputs_sheet.jpg",
        package / "model_outputs_local",
    )

    readme = package / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Film Mask Automation Review Package",
                "",
                "This folder contains the current training/test data preview and model outputs.",
                "",
                "Files:",
                "",
                "- `01_training_set_sheet.jpg`: BlueNeg training pairs used by the tiny validation model.",
                "- `02_test_set_sheet.jpg`: BlueNeg held-out test pairs.",
                "- `03_blueneg_model_outputs_sheet.jpg`: input / target / AI model output on held-out test data.",
                "- `04_local_negative_outputs_sheet.jpg`: local negative / target / manual default / manual adjusted / smart auto / tiny AI output.",
                "- `metrics.json`: MAE and parameter-sensitivity metrics.",
                "",
                "Important: `film_mask_tiny_blueneg.pt` is still a tiny CPU-trained validation checkpoint, not the final production model.",
            ]
        ),
        encoding="utf-8",
    )
    (package / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(package)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _resolve(manifest_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_root / path


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = image.convert("RGB")
    thumb.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 20, 23))
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def _make_pair_sheet(records: list[dict[str, str]], manifest_root: Path, output: Path, title: str) -> None:
    cell = (280, 190)
    header = 34
    columns = 2
    sheet = Image.new("RGB", (cell[0] * columns, (cell[1] + header) * len(records) + header), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(235, 238, 242))
    for row, record in enumerate(records):
        y = header + row * (cell[1] + header)
        negative = Image.open(_resolve(manifest_root, record["negative"]))
        target = Image.open(_resolve(manifest_root, record["target"]))
        for col, (label, image) in enumerate([("input", negative), ("target", target)]):
            draw.text((col * cell[0] + 10, y + 8), f"{row + 1} {label}", fill=(235, 238, 242))
            sheet.paste(_thumbnail(image, cell), (col * cell[0], y + header))
    sheet.save(output, quality=92)


def _make_model_output_sheet(
    records: list[dict[str, str]],
    manifest_root: Path,
    model_path: Path,
    sheet_path: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell = (280, 190)
    header = 34
    columns = 3
    sheet = Image.new("RGB", (cell[0] * columns, (cell[1] + header) * len(records) + header), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "BlueNeg test: input / target / tiny AI output", fill=(235, 238, 242))
    metrics: list[dict[str, object]] = []
    for row, record in enumerate(records):
        negative_path = _resolve(manifest_root, record["negative"])
        target_path = _resolve(manifest_root, record["target"])
        output_path = output_dir / f"test_{row + 1:02d}_ai_output.jpg"
        with Image.open(negative_path) as image:
            converted = convert_with_model(image, model_path)
            converted.save(output_path)
        metric = compare_images(output_path, target_path)
        metrics.append({"sample": row + 1, "output": str(output_path), "metrics": metric})

        y = header + row * (cell[1] + header)
        images = [
            ("input", Image.open(negative_path)),
            ("target", Image.open(target_path)),
            ("ai output", Image.open(output_path)),
        ]
        for col, (label, image) in enumerate(images):
            draw.text((col * cell[0] + 10, y + 8), f"{row + 1} {label}", fill=(235, 238, 242))
            sheet.paste(_thumbnail(image, cell), (col * cell[0], y + header))
    sheet.save(sheet_path, quality=92)
    return metrics


def _make_local_output_sheet(root: Path, model_path: Path, sheet_path: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell = (240, 160)
    header = 32
    columns = 6
    groups = [1, 2, 3]
    sheet = Image.new("RGB", (cell[0] * columns, (cell[1] + header) * len(groups) + header), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "Local negatives: negative / target / manual / adjusted / smart / tiny AI", fill=(235, 238, 242))
    metrics: list[dict[str, object]] = []

    default_params = ConversionParams(mask_source="percentile", contrast=1.0, gamma=1.0, white_balance="none")
    adjusted_params = ConversionParams(
        mask_source="percentile",
        contrast=0.78,
        gamma=1.35,
        exposure=0.35,
        brightness=0.04,
        saturation=1.2,
        temperature=0.18,
        red_gain=1.08,
        green_gain=1.0,
        blue_gain=0.92,
        white_balance="none",
    )

    for row, group in enumerate(groups):
        negative_path = root / f"{group}-1.jpg"
        target_path = root / f"{group}-2.jpg"
        with Image.open(negative_path) as negative:
            manual = convert_image(negative, default_params).image
            adjusted = convert_image(negative, adjusted_params).image
            smart = convert_image_smart(negative).image
            ai = convert_with_model(negative, model_path)
        paths = {
            "manual": output_dir / f"{group}_manual.jpg",
            "adjusted": output_dir / f"{group}_manual_adjusted.jpg",
            "smart": output_dir / f"{group}_smart.jpg",
            "ai": output_dir / f"{group}_tiny_ai.jpg",
        }
        manual.save(paths["manual"])
        adjusted.save(paths["adjusted"])
        smart.save(paths["smart"])
        ai.save(paths["ai"])

        sensitivity = _mean_abs_diff(paths["manual"], paths["adjusted"])
        metrics.append(
            {
                "group": group,
                "manual_vs_adjusted_mean_abs_diff": round(sensitivity, 2),
                "manual_metrics": compare_images(paths["manual"], target_path),
                "adjusted_metrics": compare_images(paths["adjusted"], target_path),
                "smart_metrics": compare_images(paths["smart"], target_path),
                "ai_metrics": compare_images(paths["ai"], target_path),
            }
        )

        y = header + row * (cell[1] + header)
        images = [
            ("negative", Image.open(negative_path)),
            ("target", Image.open(target_path)),
            ("manual", Image.open(paths["manual"])),
            ("adjusted", Image.open(paths["adjusted"])),
            ("smart", Image.open(paths["smart"])),
            ("tiny AI", Image.open(paths["ai"])),
        ]
        for col, (label, image) in enumerate(images):
            draw.text((col * cell[0] + 8, y + 8), f"{group} {label}", fill=(235, 238, 242))
            sheet.paste(_thumbnail(image, cell), (col * cell[0], y + header))
    sheet.save(sheet_path, quality=92)
    return metrics


def _mean_abs_diff(path_a: Path, path_b: Path) -> float:
    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
        b = np.asarray(image_b.convert("RGB").resize(image_a.size), dtype=np.float32)
    return float(np.abs(a - b).mean())


if __name__ == "__main__":
    main()
