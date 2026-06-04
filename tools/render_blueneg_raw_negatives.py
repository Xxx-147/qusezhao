from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BlueNeg DNG raw negatives into RGB PNG training inputs.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/blueneg_raw_rendered"))
    parser.add_argument("--max-size", type=int, default=1600)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()

    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("Install rawpy first: .\\.venv-ml\\Scripts\\python -m pip install rawpy") from exc

    manifest_root = args.manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "rendered_manifest.csv"
    rows: list[dict[str, str]] = []

    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        for index, row in enumerate(csv.DictReader(file), start=1):
            negative_path = _resolve(manifest_root, row["negative"])
            target_path = _resolve(manifest_root, row["target"])
            output_negative = args.output_dir / f"{index:04d}_{negative_path.stem}_negative.png"
            with rawpy.imread(str(negative_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    gamma=(1, 1),
                )
            image = Image.fromarray(rgb, mode="RGB")
            image.thumbnail((args.max_size, args.max_size))
            image.save(output_negative)
            target_value = str(Path(os.path.relpath(target_path, args.output_dir)))
            rows.append(
                {
                    "negative": str(output_negative.relative_to(args.output_dir)),
                    "target": target_value,
                    "source": row.get("source", "BlueNeg raw rendered"),
                }
            )
            print(output_negative)

    with output_manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["negative", "target", "source"])
        writer.writeheader()
        writer.writerows(rows)
    _make_pair_sheet(rows, args.output_dir, args.output_dir / "raw_negative_pair_sheet.jpg")
    print(f"wrote {len(rows)} rows to {output_manifest}")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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
    draw.text((10, 10), "BlueNeg raw DNG rendered: negative input -> pseudoGT target", fill=(235, 238, 242))
    for row_index, record in enumerate(rows):
        y = header + row_index * (cell[1] + header)
        negative = Image.open(_resolve(manifest_root, record["negative"]))
        target = Image.open(_resolve(manifest_root, record["target"]))
        labels = [f"{row_index + 1} raw negative", f"{row_index + 1} pseudoGT target"]
        for col, (label, image) in enumerate(zip(labels, [negative, target])):
            draw.text((col * cell[0] + 10, y + 8), label, fill=(235, 238, 242))
            sheet.paste(_thumbnail(image, cell), (col * cell[0], y + header))
    sheet.save(output, quality=92)
    print(output)


if __name__ == "__main__":
    main()
