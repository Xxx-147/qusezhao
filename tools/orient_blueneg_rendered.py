from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description="Orient rendered BlueNeg negatives to match their pseudoGT targets.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/blueneg_raw_rendered/rendered_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/blueneg_raw_oriented"))
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest_root = manifest_path.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for index, record in enumerate(csv.DictReader(file), start=1):
            negative_path = _resolve(manifest_root, record["negative"])
            target_path = _resolve(manifest_root, record["target"])
            with Image.open(negative_path) as negative, Image.open(target_path) as target:
                oriented, angle, score = _orient_to_target(negative.convert("RGB"), target.convert("RGB"))
            output_negative = output_dir / f"{index:04d}_{negative_path.stem}_oriented.png"
            oriented.save(output_negative)
            rows.append(
                {
                    "negative": str(output_negative.relative_to(output_dir)),
                    "target": str(Path(os.path.relpath(target_path, output_dir))),
                    "source": f"{record.get('source', 'BlueNeg')} | oriented_angle={angle} | orientation_score={score:.4f}",
                }
            )
            print(f"{index}: angle={angle} score={score:.4f} {negative_path.name}", flush=True)

    output_manifest = output_dir / "oriented_manifest.csv"
    _write_manifest(output_manifest, rows)
    _make_pair_sheet(rows, output_dir, output_dir / "oriented_pair_sheet.jpg")
    print(f"wrote {len(rows)} rows to {output_manifest}", flush=True)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _orient_to_target(negative: Image.Image, target: Image.Image) -> tuple[Image.Image, int, float]:
    candidates = [(angle, negative.rotate(angle, expand=True)) for angle in (0, 90, 180, 270)]
    target_edges = _edge_canvas(target)
    target_ratio = target.width / target.height
    best: tuple[float, int, Image.Image] | None = None
    for angle, candidate in candidates:
        candidate_ratio = candidate.width / candidate.height
        edge_score = float(np.mean(np.abs(_edge_canvas(candidate) - target_edges)))
        aspect_penalty = abs(candidate_ratio - target_ratio) / max(candidate_ratio, target_ratio)
        score = edge_score + aspect_penalty * 0.75
        if best is None or score < best[0]:
            best = (score, angle, candidate)
    assert best is not None
    return ImageOps.exif_transpose(best[2]), best[1], best[0]


def _edge_canvas(image: Image.Image, size: int = 192) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    gray.thumbnail((size, size))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(gray, ((size - gray.width) // 2, (size - gray.height) // 2))
    data = np.asarray(canvas, dtype=np.float32) / 255.0
    gx = np.abs(np.diff(data, axis=1, prepend=data[:, :1]))
    gy = np.abs(np.diff(data, axis=0, prepend=data[:1, :]))
    edges = gx + gy
    std = float(edges.std())
    if std > 1e-6:
        edges = (edges - float(edges.mean())) / std
    return edges


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["negative", "target", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = image.convert("RGB")
    thumb.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 20, 23))
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def _make_pair_sheet(rows: list[dict[str, str]], manifest_root: Path, output: Path) -> None:
    cell = (360, 236)
    header = 34
    sheet = Image.new("RGB", (cell[0] * 2, max(1, (cell[1] + header) * len(rows) + header)), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "BlueNeg oriented rendered negatives: input -> pseudoGT target", fill=(235, 238, 242))
    for row_index, record in enumerate(rows):
        y = header + row_index * (cell[1] + header)
        negative = Image.open(_resolve(manifest_root, record["negative"]))
        target = Image.open(_resolve(manifest_root, record["target"]))
        labels = [f"{row_index + 1} oriented negative", f"{row_index + 1} pseudoGT target"]
        for col, (label, image) in enumerate(zip(labels, [negative, target])):
            draw.text((col * cell[0] + 10, y + 8), label, fill=(235, 238, 242))
            sheet.paste(_thumbnail(image, cell), (col * cell[0], y + header))
    sheet.save(output, quality=92)


if __name__ == "__main__":
    main()
