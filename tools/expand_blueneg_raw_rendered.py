from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BlueNeg RAW pairs incrementally, render them, then remove DNG files.")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--repo-id", default="ttgroup/blueneg-release")
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/blueneg_raw_subset"))
    parser.add_argument("--rendered-dir", type=Path, default=Path("datasets/blueneg_raw_rendered"))
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--strategy", choices=["first", "spread"], default="spread")
    args = parser.parse_args()

    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("Install rawpy first: .\\.venv-ml\\Scripts\\python -m pip install rawpy") from exc

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: python -m pip install -e .[ml]") from exc

    raw_dir = args.raw_dir.resolve()
    rendered_dir = args.rendered_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    negative_files = [file for file in files if file.startswith("negative-16bit/") and file.endswith(".dng")]
    gt_files = [file for file in files if file.startswith("pseudogt-8bit/") and file.endswith(".pseudogt.png")]
    negatives = {_key(file, ".dng"): file for file in negative_files}
    gts = {_key(file, ".pseudogt.png"): file for file in gt_files}
    keys = _select_keys(sorted(negatives.keys() & gts.keys()), args.count, args.strategy)

    rows: list[dict[str, str]] = []
    manifest_path = rendered_dir / "rendered_manifest.csv"
    progress_path = rendered_dir / "progress.json"
    for index, key in enumerate(keys, start=1):
        expected_target_path = raw_dir / gts[key]
        negative_stem = Path(negatives[key]).stem
        output_negative = rendered_dir / f"{index:04d}_{negative_stem}_negative.png"

        if not output_negative.exists():
            negative_path = Path(hf_hub_download(args.repo_id, negatives[key], repo_type="dataset", local_dir=raw_dir))
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
            _remove_downloaded_dng(negative_path, raw_dir)
        else:
            expected_dng = raw_dir / negatives[key]
            if expected_dng.exists():
                _remove_downloaded_dng(expected_dng, raw_dir)

        target_path = expected_target_path
        if not target_path.exists():
            target_path = Path(hf_hub_download(args.repo_id, gts[key], repo_type="dataset", local_dir=raw_dir))
        rows.append(
            {
                "negative": str(output_negative.relative_to(rendered_dir)),
                "target": str(Path(os.path.relpath(target_path, rendered_dir))),
                "source": f"BlueNeg raw-negative/pseudogt: {key}",
            }
        )
        _write_manifest(manifest_path, rows)
        _write_progress(
            progress_path,
            {
                "requested_count": len(keys),
                "completed_count": len(rows),
                "current_key": key,
                "current_index": index,
                "rendered_negative": str(output_negative),
                "target": str(target_path),
                "strategy": args.strategy,
            },
        )
        print(f"{index}/{len(keys)} {key}", flush=True)

    _make_pair_sheet(rows, rendered_dir, rendered_dir / "raw_negative_pair_sheet.jpg")
    _remove_cache(raw_dir)
    print(f"wrote {len(rows)} rows to {manifest_path}", flush=True)
    print("BlueNeg requires attribution: Copyrighted by Tien-Tsin Wong", flush=True)


def _key(path: str, suffix: str) -> str:
    name = Path(path).name
    return name.removesuffix(suffix)


def _select_keys(keys: list[str], count: int, strategy: str) -> list[str]:
    if count <= 0:
        return []
    if strategy == "first" or count >= len(keys):
        return keys[:count]
    if count == 1:
        return [keys[0]]
    selected_indices = sorted({round(index * (len(keys) - 1) / (count - 1)) for index in range(count)})
    selected = [keys[index] for index in selected_indices]
    if len(selected) < count:
        used = set(selected)
        for key in keys:
            if key not in used:
                selected.append(key)
                used.add(key)
            if len(selected) == count:
                break
    return selected[:count]


def _remove_downloaded_dng(path: Path, raw_dir: Path) -> None:
    resolved = path.resolve()
    if resolved.suffix.lower() != ".dng":
        raise ValueError(f"Refusing to remove non-DNG file: {resolved}")
    if not resolved.is_relative_to(raw_dir):
        raise ValueError(f"Refusing to remove DNG outside raw directory: {resolved}")
    resolved.unlink(missing_ok=True)


def _remove_cache(raw_dir: Path) -> None:
    cache = raw_dir / ".cache"
    if cache.exists():
        resolved = cache.resolve()
        if not resolved.is_relative_to(raw_dir):
            raise ValueError(f"Refusing to remove cache outside raw directory: {resolved}")
        shutil.rmtree(resolved)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["negative", "target", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = image.convert("RGB")
    thumb.thumbnail(size)
    canvas = Image.new("RGB", size, (18, 20, 23))
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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


if __name__ == "__main__":
    main()
