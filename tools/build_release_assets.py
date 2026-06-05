from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


FIELDS = ["id", "split", "negative", "target", "source", "scanner_profile", "film_stock", "notes"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "datasets" / "mixed_true_negative_pairs" / "mixed_all_manifest.csv"
    release_root = root / "release_assets"
    dataset_root = release_root / "dataset"
    train_root = dataset_root / "train"
    test_root = dataset_root / "test"
    models_root = release_root / "models"
    experiments_root = release_root / "experiments" / "latest"

    if release_root.exists():
        shutil.rmtree(release_root)
    for path in [train_root, test_root, models_root, experiments_root]:
        path.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(manifest_path)
    output_rows: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        split = record["split"]
        split_root = train_root if split == "train" else test_root
        sample_id = f"{split}_{index:04d}"
        negative_src = _resolve(manifest_path.parent, record["negative"])
        target_src = _resolve(manifest_path.parent, record["target"])
        negative_dst = split_root / f"{sample_id}_negative{negative_src.suffix.lower()}"
        target_dst = split_root / f"{sample_id}_target{target_src.suffix.lower()}"
        shutil.copy2(negative_src, negative_dst)
        shutil.copy2(target_src, target_dst)
        output_rows.append(
            {
                "id": sample_id,
                "split": split,
                "negative": str(negative_dst.relative_to(dataset_root)).replace("\\", "/"),
                "target": str(target_dst.relative_to(dataset_root)).replace("\\", "/"),
                "source": record.get("source", ""),
                "scanner_profile": record.get("scanner_profile", ""),
                "film_stock": record.get("film_stock", ""),
                "notes": record.get("notes", ""),
            }
        )

    _write_manifest(dataset_root / "manifest.csv", output_rows)
    _write_manifest(dataset_root / "train_manifest.csv", [row for row in output_rows if row["split"] == "train"])
    _write_manifest(dataset_root / "test_manifest.csv", [row for row in output_rows if row["split"] == "test"])

    model_path = root / "models" / "film_mask_tiny_mixed_true_negative.pt"
    if model_path.exists():
        shutil.copy2(model_path, models_root / model_path.name)

    latest = root / "experiments" / "latest"
    for name in [
        "01_train_set.jpg",
        "02_test_set.jpg",
        "03_model_outputs.jpg",
        "legend.md",
        "metrics.json",
        "run_config.json",
        "sources.md",
    ]:
        source = latest / name
        if source.exists():
            shutil.copy2(source, experiments_root / name)

    summary = {
        "dataset_count": len(output_rows),
        "train_count": sum(1 for row in output_rows if row["split"] == "train"),
        "test_count": sum(1 for row in output_rows if row["split"] == "test"),
        "model": "models/film_mask_tiny_mixed_true_negative.pt",
        "manifest": "dataset/manifest.csv",
        "train_manifest": "dataset/train_manifest.csv",
        "test_manifest": "dataset/test_manifest.csv",
        "license_note": "BlueNeg images require attribution: Copyrighted by Tien-Tsin Wong. Local user-supplied samples are included with project-owner permission.",
    }
    (release_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (release_root / "README.md").write_text(_readme(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _readme(summary: dict[str, object]) -> str:
    return f"""# qusezhao release assets

This folder contains the packaged dataset, model checkpoint, and latest experiment report for the film negative mask-removal project.

## Contents

- `dataset/manifest.csv`: all paired samples
- `dataset/train_manifest.csv`: training split
- `dataset/test_manifest.csv`: test split
- `dataset/train/*_negative.*`: negative inputs
- `dataset/train/*_target.*`: target positives
- `dataset/test/*_negative.*`: held-out negative inputs
- `dataset/test/*_target.*`: held-out target positives
- `models/film_mask_tiny_mixed_true_negative.pt`: current PyTorch checkpoint
- `experiments/latest/03_model_outputs.jpg`: latest visual comparison

## Counts

- Total pairs: {summary["dataset_count"]}
- Training pairs: {summary["train_count"]}
- Test pairs: {summary["test_count"]}

## Quick Use

Install the project first:

```powershell
python -m pip install -e ".[ml]"
```

Run conversion with the included checkpoint:

```powershell
python -m film_mask_automation.cli convert input_negative.jpg output_positive.jpg --ai-model release_assets\\models\\film_mask_tiny_mixed_true_negative.pt
```

Train again using the packaged training split:

```powershell
python -m film_mask_automation.ml.train release_assets\\dataset\\train_manifest.csv models\\custom.pt --epochs 8 --steps-per-epoch 60 --batch-size 2 --crop-size 160 --base-channels 32 --device cpu
```

## Attribution

BlueNeg images require the following credit in publication, reproduction, redistribution, or derivatives:

`Copyrighted by Tien-Tsin Wong`

BlueNeg dataset: https://huggingface.co/datasets/ttgroup/blueneg-release
"""


if __name__ == "__main__":
    main()
