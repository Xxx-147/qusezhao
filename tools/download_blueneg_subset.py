from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small BlueNeg subset from Hugging Face.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--repo-id", default="ttgroup/blueneg-release")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/blueneg_subset"))
    parser.add_argument(
        "--strategy",
        choices=["first", "spread"],
        default="spread",
        help="first downloads sorted file order; spread samples evenly across all paired BlueNeg keys for more variety.",
    )
    parser.add_argument(
        "--mode",
        choices=["preview", "raw-negative"],
        default="preview",
        help="preview downloads 8-bit preview/pseudogt pairs; raw-negative downloads DNG negative/pseudogt pairs.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install ML extras first: python -m pip install -e .[ml]") from exc

    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    if args.mode == "raw-negative":
        negative_files = [file for file in files if file.startswith("negative-16bit/") and file.endswith(".dng")]
        negative_suffix = ".dng"
        manifest_path = args.output_dir / "blueneg_raw_negative_manifest.csv"
    else:
        negative_files = [file for file in files if file.startswith("negative-preview-8bit/") and file.endswith(".preview.png")]
        negative_suffix = ".preview.png"
        manifest_path = args.output_dir / "blueneg_preview_manifest.csv"

    gt_files = [file for file in files if file.startswith("pseudogt-8bit/") and file.endswith(".pseudogt.png")]
    negatives = {_key(file, negative_suffix): file for file in negative_files}
    gts = {_key(file, ".pseudogt.png"): file for file in gt_files}
    keys = _select_keys(sorted(negatives.keys() & gts.keys()), args.count, args.strategy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for key in keys:
        negative_path = Path(
            hf_hub_download(args.repo_id, negatives[key], repo_type="dataset", local_dir=args.output_dir)
        )
        gt_path = Path(hf_hub_download(args.repo_id, gts[key], repo_type="dataset", local_dir=args.output_dir))
        rows.append(
            {
                "negative": str(negative_path.relative_to(args.output_dir)),
                "target": str(gt_path.relative_to(args.output_dir)),
                "source": f"BlueNeg {args.mode}/pseudogt: {key}",
            }
        )
        print(negative_path)
        print(gt_path)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["negative", "target", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired rows to {manifest_path}")

    print("BlueNeg requires attribution: Copyrighted by Tien-Tsin Wong")


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


if __name__ == "__main__":
    main()
