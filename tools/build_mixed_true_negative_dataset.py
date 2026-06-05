from __future__ import annotations

import csv
from pathlib import Path


FIELDS = ["negative", "target", "split", "source", "scanner_profile", "film_stock", "notes"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "datasets" / "mixed_true_negative_pairs"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = discover_local_rows(root)
    rows.extend(discover_wechat_rows(root))
    rows.extend(
        [
            row(
                "../../samples/public_input/wikimedia_negative_positive_negative.jpg",
                "../../samples/public_reference/wikimedia_negative_positive_reference.jpg",
                "train",
                "wikimedia-public-domain",
                "unknown",
                "unknown",
                "public domain negative/positive split",
            ),
        ]
    )

    oriented_manifest = root / "datasets" / "blueneg_raw_oriented" / "oriented_manifest.csv"
    rendered_manifest = root / "datasets" / "blueneg_raw_rendered" / "rendered_manifest.csv"
    blueneg_manifest = oriented_manifest if oriented_manifest.exists() else rendered_manifest
    if blueneg_manifest.exists():
        with blueneg_manifest.open("r", encoding="utf-8", newline="") as file:
            for index, record in enumerate(csv.DictReader(file), start=1):
                split = "test" if index % 5 == 0 else "train"
                rows.append(
                    row(
                        str(Path("..") / ".." / "datasets" / blueneg_manifest.parent.name / record["negative"]),
                        record["target"],
                        split,
                        "BlueNeg raw DNG oriented" if blueneg_manifest == oriented_manifest else "BlueNeg raw DNG rendered",
                        "BlueNeg raw/pseudoGT",
                        "archival color negative",
                        record.get("source", "BlueNeg raw negative paired with pseudoGT"),
                    )
                )

    write(dataset_dir / "mixed_all_manifest.csv", rows)
    write(dataset_dir / "mixed_train_manifest.csv", [record for record in rows if record["split"] == "train"])
    write(dataset_dir / "mixed_test_manifest.csv", [record for record in rows if record["split"] == "test"])
    print(dataset_dir)


def row(
    negative: str,
    target: str,
    split: str,
    source: str,
    scanner_profile: str,
    film_stock: str,
    notes: str,
) -> dict[str, str]:
    return {
        "negative": negative,
        "target": target,
        "split": split,
        "source": source,
        "scanner_profile": scanner_profile,
        "film_stock": film_stock,
        "notes": notes,
    }


def discover_local_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for negative in sorted(root.glob("*-1.*")):
        prefix = negative.stem.removesuffix("-1")
        target = _first_existing(root, [f"{prefix}-2.jpg", f"{prefix}-2.png", f"{prefix}-2.tif", f"{prefix}-2.tiff"])
        if not target:
            continue
        split = _local_split(prefix)
        rows.append(
            row(
                f"../../{negative.name}",
                f"../../{target.name}",
                split,
                "local",
                "unknown",
                "unknown",
                f"user supplied true negative pair {prefix}",
            )
        )
    return rows


def discover_wechat_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    files = sorted(root.glob("微信图片_*.jpg"))
    for pair_index, index in enumerate(range(0, len(files) - 1, 2), start=1):
        negative = files[index]
        target = files[index + 1]
        split = "test" if pair_index % 4 == 0 else "train"
        rows.append(
            row(
                f"../../{negative.name}",
                f"../../{target.name}",
                split,
                "local-wechat",
                "unknown",
                "unknown",
                f"user supplied WeChat true negative pair {pair_index}: {negative.name} -> {target.name}",
            )
        )
    return rows


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _local_split(prefix: str) -> str:
    try:
        index = int(prefix)
    except ValueError:
        return "train"
    return "test" if index % 4 == 3 else "train"


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
