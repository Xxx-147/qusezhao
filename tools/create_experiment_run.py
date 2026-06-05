from __future__ import annotations

import argparse
import csv
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Create a timestamped experiment review folder.")
    parser.add_argument("--name", default="mixed_true_negative")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/mixed_true_negative_pairs/mixed_all_manifest.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/film_mask_tiny_mixed_true_negative.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    root = Path.cwd()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / args.output_root / f"{run_id}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    model_path = args.model if args.model.is_absolute() else root / args.model
    records = _read_manifest(manifest_path)
    train_records = [record for record in records if record.get("split") == "train"]
    test_records = [record for record in records if record.get("split") == "test"]

    shutil.copy2(manifest_path, run_dir / "manifest.csv")
    if model_path.exists():
        shutil.copy2(model_path, run_dir / model_path.name)

    _make_pair_sheet(train_records, manifest_path.parent, run_dir / "01_train_set.jpg", "Training set: input negative -> reference target")
    _make_pair_sheet(test_records, manifest_path.parent, run_dir / "02_test_set.jpg", "Test set: input negative -> reference target")
    metrics = {
        "train_count": len(train_records),
        "test_count": len(test_records),
        "outputs": _make_output_sheet(test_records, manifest_path.parent, model_path, run_dir / "03_model_outputs.jpg", run_dir / "outputs"),
    }

    config = {
        "run_id": run_id,
        "name": args.name,
        "manifest": str(manifest_path),
        "model": str(model_path),
        "train_count": len(train_records),
        "test_count": len(test_records),
        "note": "Each sample is paired: input negative + reference target. Test samples are not used for training.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_legend(run_dir / "legend.md")
    _write_sources(run_dir / "sources.md")

    latest = root / args.output_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    print(run_dir)


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
    sheet = Image.new("RGB", (cell[0] * 2, max(1, (cell[1] + header) * len(records) + header)), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(235, 238, 242))
    for row, record in enumerate(records):
        y = header + row * (cell[1] + header)
        negative = Image.open(_resolve(manifest_root, record["negative"]))
        target = Image.open(_resolve(manifest_root, record["target"]))
        labels = [f"{row + 1} input negative | {record.get('scanner_profile', 'unknown')}", f"{row + 1} reference target | {record.get('source', 'unknown')}"]
        for col, (label, image) in enumerate(zip(labels, [negative, target])):
            draw.text((col * cell[0] + 10, y + 8), label[:56], fill=(235, 238, 242))
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
    sheet = Image.new("RGB", (cell[0] * columns, max(1, (cell[1] + header) * len(records) + header)), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 10),
        "test: input negative / reference target / rule neutral / fixed warm / smart auto / AI hybrid",
        fill=(235, 238, 242),
    )

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
            model = convert_with_model(negative, model_path) if model_path.exists() else Image.new("RGB", negative.size, (0, 0, 0))

        output_paths = {
            "manual": output_dir / f"{row + 1:02d}_manual.jpg",
            "adjusted": output_dir / f"{row + 1:02d}_manual_adjusted.jpg",
            "smart": output_dir / f"{row + 1:02d}_smart.jpg",
            "model": output_dir / f"{row + 1:02d}_model.jpg",
        }
        manual.save(output_paths["manual"])
        adjusted.save(output_paths["adjusted"])
        smart.save(output_paths["smart"])
        model.save(output_paths["model"])
        metrics.append(
            {
                "sample": row + 1,
                "source": record.get("source"),
                "scanner_profile": record.get("scanner_profile"),
                "manual_vs_adjusted_mean_abs_diff": round(_mean_abs_diff(output_paths["manual"], output_paths["adjusted"]), 2),
                "manual_metrics": compare_images(output_paths["manual"], target_path),
                "adjusted_metrics": compare_images(output_paths["adjusted"], target_path),
                "smart_metrics": compare_images(output_paths["smart"], target_path),
                "model_metrics": compare_images(output_paths["model"], target_path),
            }
        )
        y = header + row * (cell[1] + header)
        images = [
            ("input negative", Image.open(negative_path)),
            ("reference", Image.open(target_path)),
            ("rule neutral", Image.open(output_paths["manual"])),
            ("fixed warm", Image.open(output_paths["adjusted"])),
            ("smart auto", Image.open(output_paths["smart"])),
            ("AI hybrid", Image.open(output_paths["model"])),
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


def _write_legend(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Experiment Legend",
                "",
                "| Column | Meaning | Style role |",
                "| --- | --- | --- |",
                "| input negative | Source negative scan with film mask and inverted colors. | Not a style; model input. |",
                "| reference target | Positive reference image paired with the negative. Local rows use user-supplied `*-2` files; BlueNeg rows use pseudoGT targets. | Ground truth / learning target, not generated by this project. |",
                "| rule neutral | Default rule-based conversion with percentile mask removal and no gray-world white balance. | Neutral technical baseline. |",
                "| fixed warm | A fixed hand-tuned parameter set used for comparison. | Warm/brighter baseline, not per-image optimized. |",
                "| smart auto | Automatic preset search scored by image statistics. | Batch-friendly automatic baseline. |",
                "| AI hybrid | Current checkpoint output after color restoration and smart-rule anchoring. | Learned model output stabilized by a deterministic orange-mask baseline. |",
                "",
                "These columns are comparison outputs, not final named film looks. Future style profiles should be trained or calibrated separately for targets such as Frontier SP3000, Noritsu, NLP-like, and neutral lab print.",
            ]
        ),
        encoding="utf-8",
    )


def _write_sources(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Sources",
                "",
                "- Local user-supplied true negative/positive pairs.",
                "- Wikimedia Commons Public Domain: Negative_Positive-Picture.jpg.",
                "- BlueNeg raw DNG subset from Hugging Face: https://huggingface.co/datasets/ttgroup/blueneg-release",
                "- BlueNeg paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.pdf",
                "",
                "Scanner notes:",
                "",
                "- Fujifilm Frontier SP-3000 and Noritsu scanners are important target styles, but public paired raw-negative + SP3000/Noritsu target datasets were not found in a clearly reusable form.",
                "- Future dataset rows should set `scanner_profile` to `Frontier SP3000`, `Noritsu`, or the exact lab/scanner profile whenever known.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
