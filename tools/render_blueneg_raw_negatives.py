from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


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
            try:
                target_value = str(target_path.relative_to(args.output_dir))
            except ValueError:
                target_value = str(target_path)
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
    print(f"wrote {len(rows)} rows to {output_manifest}")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
