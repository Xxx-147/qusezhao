from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image

from .model import require_torch


@dataclass(frozen=True)
class PairRecord:
    negative: Path
    target: Path
    source: str


def read_manifest(path: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    root = path.parent
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                PairRecord(
                    negative=_resolve(root, row["negative"]),
                    target=_resolve(root, row["target"]),
                    source=row.get("source", "unknown"),
                )
            )
    return records


def write_manifest(path: Path, records: list[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["negative", "target", "source"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "negative": str(record.negative),
                    "target": str(record.target),
                    "source": record.source,
                }
            )


def discover_local_pairs(root: Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    for negative in sorted(root.glob("*-1.*")):
        stem_prefix = negative.stem.removesuffix("-1")
        target = _first_existing(root, [f"{stem_prefix}-2.jpg", f"{stem_prefix}-2.png", f"{stem_prefix}-2.tif"])
        if target:
            records.append(PairRecord(negative=negative, target=target, source="local-numbered-pair"))

    public_negative = root / "samples" / "public_input" / "wikimedia_negative_positive_negative.jpg"
    public_target = root / "samples" / "public_reference" / "wikimedia_negative_positive_reference.jpg"
    if public_negative.exists() and public_target.exists():
        records.append(PairRecord(public_negative, public_target, "wikimedia-public-domain"))
    return records


class NegativePairDataset:
    def __init__(self, manifest_path: Path, crop_size: int = 256, augment: bool = True) -> None:
        self.records = read_manifest(manifest_path)
        self.crop_size = crop_size
        self.augment = augment
        self.torch = require_torch()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        record = self.records[index % len(self.records)]
        negative = Image.open(record.negative).convert("RGB")
        target = Image.open(record.target).convert("RGB").resize(negative.size)
        negative, target = _paired_crop(negative, target, self.crop_size)
        if self.augment and random.random() < 0.5:
            negative = negative.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return _to_tensor(negative, self.torch), _to_tensor(target, self.torch)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _paired_crop(negative: Image.Image, target: Image.Image, crop_size: int) -> tuple[Image.Image, Image.Image]:
    width, height = negative.size
    size = min(crop_size, width, height)
    if width == size and height == size:
        return negative, target
    left = random.randint(0, max(width - size, 0))
    top = random.randint(0, max(height - size, 0))
    box = (left, top, left + size, top + size)
    return negative.crop(box), target.crop(box)


def _to_tensor(image: Image.Image, torch):  # type: ignore[no-untyped-def]
    import numpy as np

    array = np.asarray(image, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
