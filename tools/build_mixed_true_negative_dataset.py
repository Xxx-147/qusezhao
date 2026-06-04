from __future__ import annotations

import csv
from pathlib import Path


FIELDS = ["negative", "target", "split", "source", "scanner_profile", "film_stock", "notes"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "datasets" / "mixed_true_negative_pairs"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row("../../1-1.jpg", "../../1-2.jpg", "train", "local", "unknown", "unknown", "user supplied true negative pair 1"),
        row("../../2-1.jpg", "../../2-2.jpg", "train", "local", "unknown", "unknown", "user supplied true negative pair 2"),
        row(
            "../../samples/public_input/wikimedia_negative_positive_negative.jpg",
            "../../samples/public_reference/wikimedia_negative_positive_reference.jpg",
            "train",
            "wikimedia-public-domain",
            "unknown",
            "unknown",
            "public domain negative/positive split",
        ),
        row("../../3-1.jpg", "../../3-2.jpg", "test", "local", "unknown", "unknown", "user supplied held-out true negative pair 3"),
    ]

    blueneg_manifest = root / "datasets" / "blueneg_raw_rendered" / "rendered_manifest.csv"
    if blueneg_manifest.exists():
        with blueneg_manifest.open("r", encoding="utf-8", newline="") as file:
            for index, record in enumerate(csv.DictReader(file), start=1):
                split = "test" if index % 5 == 0 else "train"
                rows.append(
                    row(
                        f"../../datasets/blueneg_raw_rendered/{record['negative']}",
                        record["target"],
                        split,
                        "BlueNeg raw DNG rendered",
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


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
