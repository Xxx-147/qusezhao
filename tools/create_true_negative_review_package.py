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
    parser = argparse.ArgumentParser(description="Create review package for true negative-to-positive pairs.")
    parser.add_argument("--output-dir", type=Path, default=Path("review_packages/2026-06-04_true_negative_review"))
    parser.add_argument("--model", type=Path, default=Path("models/film_mask_tiny_true_negative.pt"))
    args = parser.parse_args()

    root = Path.cwd()
    package = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)

    dataset_dir = root / "datasets" / "true_negative_pairs"
    train_manifest = dataset_dir / "true_train_manifest.csv"
    test_manifest = dataset_dir / "true_test_manifest.csv"
    all_manifest = dataset_dir / "true_all_manifest.csv"
    train_records = _read_manifest(train_manifest)
    test_records = _read_manifest(test_manifest)
    all_records = _read_manifest(all_manifest)

    shutil.copy2(train_manifest, package / "true_train_manifest.csv")
    shutil.copy2(test_manifest, package / "true_test_manifest.csv")
    shutil.copy2(all_manifest, package / "true_all_manifest.csv")

    _make_pair_sheet(train_records, dataset_dir, package / "01_true_training_set.jpg", "True training set: orange negative -> target positive")
    _make_pair_sheet(test_records, dataset_dir, package / "02_true_test_set.jpg", "True held-out test set: orange negative -> target positive")

    metrics: dict[str, object] = {}
    metrics["true_test_outputs"] = _make_output_sheet(
        test_records,
        dataset_dir,
        args.model,
        package / "03_true_test_outputs.jpg",
        package / "outputs_true_test",
    )
    metrics["all_true_pair_outputs"] = _make_output_sheet(
        all_records,
        dataset_dir,
        args.model,
        package / "04_all_true_pair_outputs.jpg",
        package / "outputs_all_true_pairs",
    )

    (package / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (package / "README.md").write_text(
        "\n".join(
            [
                "# True Negative Review Package",
                "",
                "This package replaces the previous BlueNeg 8-bit preview package.",
                "",
                "It only contains real orange negative / target positive pairs:",
                "",
                "- local `1-1 -> 1-2`",
                "- local `2-1 -> 2-2`",
                "- local `3-1 -> 3-2` held out as test",
                "- Wikimedia Commons Public Domain `Negative_Positive-Picture.jpg` split into negative/positive",
                "",
                "The AI checkpoint is still a tiny validation model trained from only 3 true pairs. It proves the pipeline, not final quality.",
            ]
        ),
        encoding="utf-8",
    )
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
    cell = (360, 236)
    header = 34
    sheet = Image.new("RGB", (cell[0] * 2, (cell[1] + header) * len(records) + header), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(235, 238, 242))
    for row, record in enumerate(records):
        y = header + row * (cell[1] + header)
        negative = Image.open(_resolve(manifest_root, record["negative"]))
        target = Image.open(_resolve(manifest_root, record["target"]))
        for col, (label, image) in enumerate([("orange negative input", negative), ("target positive", target)]):
            draw.text((col * cell[0] + 10, y + 8), f"{row + 1} {label}", fill=(235, 238, 242))
            sheet.paste(_thumbnail(image, cell), (col * cell[0], y + header))
    sheet.save(output, quality=92)


def _make_output_sheet(
    records: list[dict[str, str]],
    manifest_root: Path,
    model_path: Path,
    sheet_path: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell = (260, 174)
    header = 34
    columns = 6
    sheet = Image.new("RGB", (cell[0] * columns, (cell[1] + header) * len(records) + header), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "negative / target / manual / adjusted / smart / tiny AI", fill=(235, 238, 242))

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
        blue_gain=0.92,
        white_balance="none",
    )

    metrics: list[dict[str, object]] = []
    for row, record in enumerate(records):
        negative_path = _resolve(manifest_root, record["negative"])
        target_path = _resolve(manifest_root, record["target"])
        with Image.open(negative_path) as negative:
            manual = convert_image(negative, default_params).image
            adjusted = convert_image(negative, adjusted_params).image
            smart = convert_image_smart(negative).image
            ai = convert_with_model(negative, model_path)

        output_paths = {
            "manual": output_dir / f"{row + 1:02d}_manual.jpg",
            "adjusted": output_dir / f"{row + 1:02d}_manual_adjusted.jpg",
            "smart": output_dir / f"{row + 1:02d}_smart.jpg",
            "ai": output_dir / f"{row + 1:02d}_tiny_ai.jpg",
        }
        manual.save(output_paths["manual"])
        adjusted.save(output_paths["adjusted"])
        smart.save(output_paths["smart"])
        ai.save(output_paths["ai"])

        metrics.append(
            {
                "sample": row + 1,
                "source": record["source"],
                "manual_vs_adjusted_mean_abs_diff": round(_mean_abs_diff(output_paths["manual"], output_paths["adjusted"]), 2),
                "manual_metrics": compare_images(output_paths["manual"], target_path),
                "adjusted_metrics": compare_images(output_paths["adjusted"], target_path),
                "smart_metrics": compare_images(output_paths["smart"], target_path),
                "ai_metrics": compare_images(output_paths["ai"], target_path),
            }
        )

        y = header + row * (cell[1] + header)
        images = [
            ("negative", Image.open(negative_path)),
            ("target", Image.open(target_path)),
            ("manual", Image.open(output_paths["manual"])),
            ("adjusted", Image.open(output_paths["adjusted"])),
            ("smart", Image.open(output_paths["smart"])),
            ("tiny AI", Image.open(output_paths["ai"])),
        ]
        for col, (label, image) in enumerate(images):
            draw.text((col * cell[0] + 8, y + 8), f"{row + 1} {label}", fill=(235, 238, 242))
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
